"""Custom decoder definition for transducer models."""

import torch

from espnet.nets.pytorch_backend.transducer.blocks import build_blocks
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.pytorch_backend.transducer.utils import check_state
from espnet.nets.pytorch_backend.transducer.utils import pad_batch_state
from espnet.nets.pytorch_backend.transducer.utils import pad_sequence
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.transducer_decoder_interface import TransducerDecoderInterface


class CustomDecoder(TransducerDecoderInterface, torch.nn.Module):
    """Custom decoder module for transducer models.

    Args:
        odim (int): dimension of outputs
        edim (int): dimension of encoder outputs
        jdim (int): dimension of joint-space
        dec_arch (list): list of layer definitions
        input_layer (str): input layer type
        repeat_block (int): repeat provided blocks N times if N > 1
        joint_activation_type (str) joint network activation type
        positional_encoding_type (str): positional encoding type
        positionwise_layer_type (str): linear
        positionwise_activation_type (str): positionwise activation type
        dropout_rate_embed (float): dropout rate for embedding layer (if specified)
        blank (int): blank symbol ID

    """

    def __init__(
        self,
        odim,
        edim,
        jdim,
        dec_arch,
        input_layer="embed",
        repeat_block=0,
        joint_activation_type="tanh",
        positional_encoding_type="abs_pos",
        positionwise_layer_type="linear",
        positionwise_activation_type="relu",
        dropout_rate_embed=0.0,
        blank=0,
    ):
        """Construct a CustomDecoder object."""
        torch.nn.Module.__init__(self)

        self.embed, self.decoders, ddim, _ = build_blocks(
            "decoder",
            odim,
            input_layer,
            dec_arch,
            repeat_block=repeat_block,
            positional_encoding_type=positional_encoding_type,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_activation_type=positionwise_activation_type,
            dropout_rate_embed=dropout_rate_embed,
            padding_idx=blank,
        )

        self.after_norm = LayerNorm(ddim)

        self.joint_network = JointNetwork(odim, edim, ddim, jdim, joint_activation_type)

        self.dunits = ddim
        self.odim = odim

        self.blank = blank

    def init_state(self, batch_size=None, device=None, dtype=None):
        """Initialize decoder states.

        Args:
            init_tensor (torch.Tensor): batch of input features (B, dec_dim)

        Returns:
            state (list): batch of decoder decoder states [L x None]

        """
        state = [None] * len(self.decoders)

        return state

    def forward(self, tgt, tgt_mask, memory):
        """Forward custom decoder.

        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out)
                                if input_layer == "embed"
                                input tensor
                                (batch, maxlen_out, #mels) in the other cases
            tgt_mask (torch.Tensor): input token mask,  (batch, maxlen_out)
                                     dtype=torch.uint8 in PyTorch 1.2-
                                     dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory (torch.Tensor): encoded memory, float32  (batch, maxlen_in, feat)

        Return:
            z (torch.Tensor): joint output (batch, maxlen_in, maxlen_out, odim)
            tgt_mask (torch.Tensor): score mask before softmax (batch, maxlen_out)

        """
        tgt = self.embed(tgt)

        tgt, tgt_mask = self.decoders(tgt, tgt_mask)
        tgt = self.after_norm(tgt)

        h_enc = memory.unsqueeze(2)
        h_dec = tgt.unsqueeze(1)

        z = self.joint_network(h_enc, h_dec)

        return z, tgt_mask

    def score(self, hyp, cache):
        """Forward one step.

        Args:
            hyp (dataclass): hypothesis
            cache (dict): states cache

        Returns:
            y (torch.Tensor): decoder outputs (1, dec_dim)
            (list): decoder states
                [L x (1, max_len, dec_dim)]
            lm_tokens (torch.Tensor): token id for LM (1)

        """
        device = next(self.parameters()).device

        tgt = torch.tensor(hyp.yseq).unsqueeze(0).to(device=device)
        lm_tokens = tgt[:, -1]

        str_yseq = "".join([str(x) for x in hyp.yseq])

        if str_yseq in cache:
            y, new_state = cache[str_yseq]
        else:
            tgt_mask = subsequent_mask(len(hyp.yseq)).unsqueeze(0).to(device=device)

            state = check_state(hyp.dec_state, (tgt.size(1) - 1), self.blank)

            tgt = self.embed(tgt)

            new_state = []
            for s, decoder in zip(state, self.decoders):
                tgt, tgt_mask = decoder(tgt, tgt_mask, cache=s)
                new_state.append(tgt)

            y = self.after_norm(tgt[:, -1])

            cache[str_yseq] = (y, new_state)

        return y[0], new_state, lm_tokens

    def batch_score(self, hyps, batch_states, cache):
        """Forward batch one step.

        Args:
            hyps (list): batch of hypotheses
            batch_states (list): decoder states
                [L x (B, max_len, dec_dim)]
            cache (dict): states cache

        Returns:
            batch_y (torch.Tensor): decoder output (B, dec_dim)
            batch_states (list): decoder states
                [L x (B, max_len, dec_dim)]
            lm_tokens (torch.Tensor): batch of token ids for LM (B)

        """
        final_batch = len(hyps)
        device = next(self.parameters()).device

        process = []
        done = [None for _ in range(final_batch)]

        for i, hyp in enumerate(hyps):
            str_yseq = "".join([str(x) for x in hyp.yseq])

            if str_yseq in cache:
                done[i] = (*cache[str_yseq], hyp.yseq)
            else:
                process.append((str_yseq, hyp.yseq, hyp.dec_state))

        if process:
            batch = len(process)
            _tokens = pad_sequence([p[1] for p in process], self.blank)
            _states = [p[2] for p in process]

            batch_tokens = torch.LongTensor(_tokens).view(batch, -1).to(device=device)
            tgt_mask = (
                subsequent_mask(batch_tokens.size(-1))
                .unsqueeze(0)
                .expand(batch, -1, -1)
                .to(device=device)
            )

            dec_state = self.init_state()
            dec_state = self.create_batch_states(
                dec_state,
                _states,
                _tokens,
            )

            tgt = self.embed(batch_tokens)

            next_state = []
            for s, decoder in zip(dec_state, self.decoders):
                tgt, tgt_mask = decoder(tgt, tgt_mask, cache=s)
                next_state.append(tgt)

            tgt = self.after_norm(tgt[:, -1])

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                new_state = self.select_state(next_state, j)

                done[i] = (tgt[j], new_state, process[j][2])
                cache[process[j][0]] = (tgt[j], new_state)

                j += 1

        batch_states = self.create_batch_states(
            batch_states, [d[1] for d in done], [d[2] for d in done]
        )
        batch_y = torch.stack([d[0] for d in done])

        lm_tokens = (
            torch.LongTensor([hyp.yseq[-1] for hyp in hyps])
            .view(final_batch)
            .to(device=device)
        )

        return batch_y, batch_states, lm_tokens

    def select_state(self, batch_states, idx):
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (list): batch of decoder states
                [L x (B, max_len, dec_dim)]
            idx (int): index to extract state from batch of states

        Returns:
            state_idx (list): decoder states for given id
                [L x (1, max_len, dec_dim)]

        """
        if batch_states[0] is None:
            return batch_states

        state_idx = [batch_states[layer][idx] for layer in range(len(self.decoders))]

        return state_idx

    def create_batch_states(self, batch_states, l_states, l_tokens):
        """Create batch of decoder states.

        Args:
            batch_states (list): batch of decoder states
                [L x (B, max_len, dec_dim)]
            l_states (list): list of decoder states
                [B x [L x (1, max_len, dec_dim)]]
            l_tokens (list): list of token sequences for batch

        Returns:
            batch_states (list): batch of decoder states
                [L x (B, max_len, dec_dim)]

        """
        if batch_states[0] is None:
            return batch_states

        max_len = max([len(t) for t in l_tokens])

        for layer in range(len(self.decoders)):
            batch_states[layer] = pad_batch_state(
                [s[layer] for s in l_states], max_len, self.blank
            )

        return batch_states
