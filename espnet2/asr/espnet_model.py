from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import editdistance

import torch
from torch.nn.utils.rnn import pad_sequence
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.batch_beam_search_batch import BatchBeamSearchBatch
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        ctc: CTC,
        rnnt_decoder: None,
        ctc_weight: float = 0.5,
        mwer_weight: float = 0.0,
        beam_size: int = 5,
        nbest: int = 5,
        nbest_ctc_scale: float = 0.0,
        nbest_penalty: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.decoder = decoder
        self.mwer_weight = mwer_weight
        self.beam_size = beam_size
        self.nbest = nbest
        self.nbest_ctc_scale = nbest_ctc_scale
        self.nbest_penalty = nbest_penalty
        self.length_normalized_loss = length_normalized_loss
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # generate nbest and nbest_data for mwer training
        if self.training and self.mwer_weight != 0:
            # do beam search
            self.eval()
            nbest_hyps = self._beam_search(speech, speech_lengths)
            # keep nbest hypothesis
            nbest_hyps = [nbest_hyps[h][: self.nbest] for h in range(len(nbest_hyps))]
            hyp_scores = [[nbest_hyps[h][n].score for n in range(self.nbest)] for h in range(len(nbest_hyps))]
            hyp_scores = torch.tensor(hyp_scores).to(text.device)
            hyp_scores = hyp_scores.exp()
            speech, speech_lengths, text, text_lengths = self._make_mwer_data(
                nbest_hyps, speech, speech_lengths, text, text_lengths)
            self.train()

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        elif self.mwer_weight == 0.0 or not self.training:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_mwer_loss(
                encoder_out, encoder_out_lens, text, text_lengths, batch_size, hyp_scores
            )

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation for spectrogram
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_att_mwer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int,
        hyp_scores: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute ce loss
        decoder_out_true = decoder_out[:batch_size]
        ys_out_pad_true = ys_out_pad[:batch_size]
        loss_ce = self.criterion_att(decoder_out_true, ys_out_pad_true)

        # 3. Compute mwer loss
        decoder_out_nbest = decoder_out[batch_size:]
        ys_out_pad_nbest = ys_out_pad[batch_size:]
        loss_mwer = self._mwer_loss(decoder_out_nbest, ys_out_pad_nbest, ys_out_pad_true, hyp_scores)
        
        loss = self.mwer_weight * loss_mwer + (1 - self.mwer_weight) * loss_ce
        
        # Compute acc and cer/wer using attention-decoder
        acc_att = th_accuracy(
            decoder_out_true.view(-1, self.vocab_size),
            ys_out_pad_true,
            ignore_label=self.ignore_id
        )
        
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out_true.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad[:batch_size].cpu())

        return loss, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError

    def _beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        token_type: str = None,
        bpemodel: str = None,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        dtype: str = "float32",
    ):

        # Initialization
        # Build BeamSearch object
        scorers = {}
        decoder = self.decoder
        ctc = CTCPrefixScorer(ctc=self.ctc, eos=self.eos)   
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(self.token_list))
        )   

        weights = dict(
            decoder=1.0 - self.nbest_ctc_scale,
            ctc=self.nbest_ctc_scale,
            length_bonus=self.nbest_penalty,
        )

        beam_search = BeamSearch(
            beam_size=self.beam_size,
            weights=weights,
            scorers=scorers,
            sos=self.sos,
            eos=self.eos,
            vocab_size=len(self.token_list),
            token_list=self.token_list,
            pre_beam_score_key=None if self.nbest_ctc_scale == 1.0 else "full",
        )

        # All scorers should support batch computation
        for k, v in beam_search.full_scorers.items():
            assert isinstance(v, BatchScorerInterface), type(v) 
        beam_search.__class__ = BatchBeamSearchBatch

        # Hardcode, mwer only support gpu training
        device = "cuda"
        beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
       
        self.eval()
        for scorer in scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
         
        # [Optional] Build Text converter: e.g. bpe-sys -> Text
        # TODO(Wangzhcihao): support all token_type
        if token_type is None:
            token_type = "char"
        if bpemodel is None:
            pass
               
        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=self.token_list) 

        # Begin beam search
        batch = {"speech": speech, "speech_lengths": speech_lengths}
        enc, ilens = self.encode(**batch)
        nbest_hyps = beam_search(x=enc, ilens=ilens, maxlenratio=maxlenratio, minlenratio=minlenratio)

        return nbest_hyps

    def _make_mwer_data(
        self,
        nbest_hyps: List[List[Hypothesis]],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        batch_size = speech.shape[0]
        assert batch_size == len(nbest_hyps), len(nbest_hyps)
        nbest_yseq_list = [nbest_hyps[h][n].yseq[1:-1] for h in range(len(nbest_hyps)) 
            for n in range(self.nbest)]

        # combine ori and nbest text
        nbest_lengths = torch.tensor([len(nbest_yseq_list[i]) 
            for i in range(len(nbest_yseq_list))]).to(text_lengths.device)
        text_lengths = torch.cat((text_lengths, nbest_lengths), dim=0)

        nbest_text = pad_sequence(nbest_yseq_list, batch_first=True, padding_value=self.ignore_id)
        assert len(nbest_text) == batch_size * self.nbest, len(nbest_text)

        # make sure text and nbest_text have the same dim
        if text.shape[1] < nbest_text.shape[1]:
            text_pad = torch.full((text.shape[0], nbest_text.shape[1] - text.shape[1]), self.ignore_id, dtype=text.dtype).to(text.device)
            text = torch.cat((text, text_pad), dim=1)
        elif text.shape[1] > nbest_text.shape[1]:
            nbest_pad = torch.full((nbest_text.shape[0], text.shape[1] - nbest_text.shape[1]), self.ignore_id, dtype=text.dtype).to(text.device)
            nbest_text = torch.cat((nbest_text, nbest_pad), dim=1)

        text = torch.cat((text, nbest_text), dim=0)
     
        # combine ori and nbest speech
        nbest_speech = speech.unsqueeze(1).repeat(1, self.nbest, 1, 1).contiguous()
        nbest_speech = nbest_speech.view(batch_size * self.nbest, speech.shape[1], speech.shape[2])
        nbest_speech_lengths = speech_lengths.repeat(self.nbest).view(self.nbest, batch_size).transpose(0, 1).contiguous()
        nbest_speech_lengths = nbest_speech_lengths.view(-1)

        speech = torch.cat((speech, nbest_speech), dim=0)
        speech_lengths = torch.cat((speech_lengths, nbest_speech_lengths), dim=0)

        return speech, speech_lengths, text, text_lengths

    def _mwer_loss(
        self,
        decoder_output_nbest: torch.Tensor,
        ys_out_pad_nbest: torch.Tensor,
        label: torch.Tensor,
        hyp_scores: torch.Tensor,
    ):
        """
        Args:
            decoder_output_nbest: (batch*nbest, maxLen, class)
            ys_out_pad_nbest: (batch*nbest, maxLen)
            label:(batch, maxLen)
        """
        batch_size = label.shape[0]
        mask_lab = label != self.ignore_id
        decoder_output_nbest = decoder_output_nbest.contiguous().view(batch_size, self.nbest, label.shape[1], -1)
        ys_out_pad_nbest = ys_out_pad_nbest.contiguous().view(batch_size, self.nbest, -1)
        ignore = ys_out_pad_nbest == self.ignore_id
        ys_nbest_index = ys_out_pad_nbest.masked_fill(ignore, 0)  # avoid -1 index
        select = torch.gather(decoder_output_nbest, -1, ys_nbest_index.unsqueeze(-1)).squeeze(-1).masked_fill(ignore != 0, 0)
        scores = torch.sum(select, dim=-1)
        # TODO(wangzhichao):use score in beam-search
        hyp_scores = torch.nn.functional.softmax(hyp_scores, dim=1)
        loss = torch.nn.functional.softmax(scores * hyp_scores, dim=1)
    
        # Compoute batch cer
        ys_out_pad_nbest = ys_out_pad_nbest.contiguous().view(batch_size, self.nbest, -1)
        label_host = label.cpu().numpy().reshape(batch_size, -1)
        nbest_host = ys_out_pad_nbest.cpu().numpy()
        error = torch.zeros(batch_size, self.nbest)

        for i in range(batch_size):
            lab_i = [idx for idx in label_host[i] if idx != self.ignore_id] 
            for j in range(self.nbest):
                hyp_j = [idx for idx in nbest_host[i][j] if idx != self.ignore_id]
                error[i][j] = editdistance.eval(hyp_j, lab_i)
                
        error = error.to(loss.device)
        denom = mask_lab.sum().item() if self.length_normalized_loss else batch_size
        loss = torch.sum(loss * error) / denom

        return loss 
