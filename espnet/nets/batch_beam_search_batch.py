"""Parallel beam search module."""

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from espnet.nets.beam_search import BeamSearch
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.beam_search import Hypothesis


class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""

    yseq: torch.Tensor  # (batch * beam, maxlen)
    score: torch.Tensor  # (batch, beam)
    length: torch.Tensor  # (batch * beam)
    scores: Dict[str, torch.Tensor] = dict()  # values: (batch, beam)
    states: Dict[str, Dict] = dict()

    def __len__(self) -> int:
        """Return a batch size."""
        return len(self.length)


class BatchBeamSearchBatch(BeamSearch):
    """Batch beam search implementation."""

    def batchfy(self, hyps: List[Hypothesis]) -> BatchHypothesis:
        """Convert list to batch."""
        if len(hyps) == 0:
            return BatchHypothesis()
        return BatchHypothesis(
            yseq=pad_sequence(
                [h.yseq for h in hyps], batch_first=True, padding_value=self.eos
            ),
            length=torch.tensor([len(h.yseq) for h in hyps], dtype=torch.int64),
            score=torch.tensor([h.score for h in hyps]),
            scores={k: torch.tensor([h.scores[k] for h in hyps]) for k in self.scorers},
            states={k: [h.states[k] for h in hyps] for k in self.scorers},
        )
   
    def _batch_reset(self, hyps: BatchHypothesis, ids: List[int]) -> BatchHypothesis:
        #hyps.score[ids] = 0.0
        hyps.length[ids] = 0
        return hyps

    def _batch_select(self, hyps: BatchHypothesis, ids: List[int]) -> BatchHypothesis:
        return BatchHypothesis(
            yseq=hyps.yseq[ids],
            score=hyps.score[ids],
            length=hyps.length[ids],
            scores={k: v[ids] for k, v in hyps.scores.items()},
            states={
                k: [self.scorers[k].select_state(v, i) for i in ids]
                for k, v in hyps.states.items()
            },
        )

    def _select(self, hyps: BatchHypothesis, i: int) -> Hypothesis:
        return Hypothesis(
            yseq=hyps.yseq[i, : hyps.length[i]],
            score=hyps.score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            states={
                k: self.scorers[k].select_state(v, i) for k, v in hyps.states.items()
            },
        )

    def unbatchfy(self, batch_hyps: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        return [
            Hypothesis(
                yseq=batch_hyps.yseq[i][: batch_hyps.length[i]],
                score=batch_hyps.score[i],
                scores={k: batch_hyps.scores[k][i] for k in self.scorers},
                states={
                    k: v.select_state(batch_hyps.states[k], i)
                    for k, v in self.scorers.items()
                },
            )
            for i in range(len(batch_hyps.length))
        ]

    def batch_beam(
        self, weighted_scores: torch.Tensor, ids: torch.Tensor, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-compute topk full token ids and partial token ids.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
                Its shape is `(batch * n_beam, self.vocab_size)`.
            ids (torch.Tensor): The partial token ids to compute topk.
                Its shape is `(batch_size * n_beam, self.pre_beam_size)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The topk full (prev_hyp, new_token) ids
                and partial (prev_hyp, new_token) ids.
                Their shapes are all `(self.beam_size,)`

        """
        top_ids = weighted_scores.view(batch_size, -1).topk(self.beam_size)[1]
        # Because of the flatten above, `top_ids` is organized as:
        # [hyp1 * V + token1, hyp2 * V + token2, ..., hypK * V + tokenK],
        # where V is `self.n_vocab` and K is `self.beam_size`
        index_batch = (torch.arange(batch_size) * self.beam_size).view(-1, 1).to(weighted_scores.device)
        prev_hyp_ids = (top_ids // self.n_vocab + index_batch).view(-1)
        new_token_ids = (top_ids % self.n_vocab).view(-1)
        return prev_hyp_ids, new_token_ids, prev_hyp_ids, new_token_ids

    def init_hyp(self, x: torch.Tensor) -> BatchHypothesis:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature

        Returns:
            Hypothesis: The initial hypothesis.

        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.batch_init_state(x)
            init_scores[k] = 0.0
        return self.batchfy(
            [
                Hypothesis(
                    score=0.0,
                    scores=init_scores,
                    states=init_states,
                    yseq=torch.tensor([self.sos], device=x.device),
                ) for _ in range(x.shape[0])
            ]
        )

    def score_full(
        self, hyp: BatchHypothesis, x: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature
            ilens (torch.Tensor): Input feature length

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            scores[k], states[k] = d.batch_score_batch(hyp.yseq, hyp.length, hyp.states[k], x, ilens)
        return scores, states

    def score_partial(
        self, hyp: BatchHypothesis, ids: torch.Tensor, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            ids (torch.Tensor): 2D tensor of new partial tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.batch_score_partial(
                hyp.yseq, ids, hyp.states[k], x
            )
        return scores, states

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        """Merge states for new hypothesis.

        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.

        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, v in part_states.items():
            new_states[k] = v
        return new_states

    def search(self, running_hyps: BatchHypothesis, x: torch.Tensor, ilens: torch.Tensor) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (Batch * beam, T, D)
            ilens (torch.Tensor): Encoded speech length (Batch * beam,)

        Returns:
            BatchHypothesis: Best sorted hypotheses

        """
        n_batch = len(running_hyps)
        batch_size = n_batch // self.beam_size
        part_ids = None  # no pre-beam
        # batch scoring
        weighted_scores = torch.zeros(
            n_batch, self.n_vocab, dtype=x.dtype, device=x.device
        )
        scores, states = self.score_full(running_hyps, x, ilens)
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]

        # partial scoring
        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]
        # NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
        # full-size score matrices, which has non-zero scores for part_ids and zeros
        # for others.
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x)
  
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]
        # add previous hyp scores
        weighted_scores += running_hyps.score.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)

        # mask weighted_scores
        score_mask = running_hyps.length.eq(0).unsqueeze(1).to(weighted_scores.device)
        weighted_scores = weighted_scores.masked_fill(score_mask, self.logzero)
        # TODO(karita): do not use list. use batch instead
        # see also https://github.com/espnet/espnet/pull/1402#discussion_r354561029
        # update hyps
        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)

        # handle the first token
        if running_hyps.length[0] == 1:
            batch_index = torch.arange(batch_size) * self.beam_size 
            search_scores = weighted_scores[batch_index]
        else:
            search_scores = weighted_scores

        for (
            full_prev_hyp_id,
            full_new_token_id,
            part_prev_hyp_id,
            part_new_token_id,
        ) in zip(*self.batch_beam(search_scores, part_ids, batch_size)):
            prev_hyp = prev_hyps[full_prev_hyp_id]
            best_hyps.append(
                Hypothesis(
                    score=weighted_scores[full_prev_hyp_id, full_new_token_id],
                    yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
                    scores=self.merge_scores(
                        prev_hyp.scores,
                        {k: v[full_prev_hyp_id] for k, v in scores.items()},
                        full_new_token_id,
                        {k: v[part_prev_hyp_id] for k, v in part_scores.items()},
                        part_new_token_id,
                    ),
                    states=self.merge_states(
                        {
                            k: self.full_scorers[k].select_state(v, full_prev_hyp_id)
                            for k, v in states.items()
                        },
                        {
                            k: self.part_scorers[k].select_state(
                                v, part_prev_hyp_id, part_new_token_id
                            )
                            for k, v in part_states.items()
                        },
                        part_new_token_id,
                    ),
                )
            )
        return self.batchfy(best_hyps)

    def forward(
        self, x: torch.Tensor, ilens: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (B, T, D)
            ilens (torch.Tensor): Encoded speech length (B,)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        # set length bounds
        batch_size = len(ilens)
        if maxlenratio == 0:
            maxlen = int(max(ilens))
        else:
            maxlen = max(1, int(maxlenratio * max(ilens)))
        minlen = int(minlenratio * maxlen)
        logging.info(f"decoder input length: {ilens}")
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))
        logging.info("output lengths:" + str(ilens))
        
        # expand x to exp_x: (Batch*beam, T, D)
        exp_x = x.unsqueeze(1).repeat(1, self.beam_size, 1, 1).contiguous()
        exp_x = exp_x.view(batch_size*self.beam_size, x.size()[1], x.size()[2])

        # expand ilens to exp_ilens(Batch*beam, )
        exp_ilens = ilens.repeat(self.beam_size).view(self.beam_size, batch_size).transpose(0, 1).contiguous()
        exp_ilens = exp_ilens.view(-1)        

        # main loop of prefix search
        running_hyps = self.init_hyp(exp_x)
        ended_hyps = [[] for _ in range(batch_size)]
        stop_search = [ False for _ in range(batch_size)]
        for i in range(maxlen):
            logging.debug("position " + str(i))
            best = self.search(running_hyps, exp_x, exp_ilens)
            # post process of one iteration
            running_hyps = self.post_process(i, ilens, maxlenratio, best, ended_hyps, stop_search)
            # end detection
            for utt_i in range(batch_size):
                if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps[utt_i]], i):
                    logging.info(f"end detected at {i} in utterance {utt_i}")
                    stop_search[utt_i] = True
                    end_ids = [ids for ids in range(utt_i * self.beam_size, (utt_i + 1) * self.beam_size)]
                    running_hyps = self._batch_reset(running_hyps, end_ids)
            if running_hyps.length.sum() == 0:
                logging.info("no hypothesis. Finish decoding.")
                break
            else:
                logging.debug(f"remained hypotheses: {len(torch.nonzero(running_hyps.length).view(-1))}")

        nbest_hyps = [sorted(ended_hyps[utt_i], key=lambda x: x.score, reverse=True) for utt_i in range(batch_size)]
        # check the number of hypotheses reaching to eos
        nbest_num = [len(nbest_hyps[utt_i]) for utt_i in range(batch_size)]
        if 0 in nbest_num:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                nbest_hyps
                if minlenratio < 0.1
                else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
            )
        # report the best result
        for utt_i in range(batch_size):
            best = nbest_hyps[utt_i][0]
            for k, v in best.scores.items():
                logging.info(
                    f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
                )
            logging.info(f"total log probability: {best.score:.2f}")
            logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
            logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
            if self.token_list is not None:
                logging.info(
                    "best hypo: "
                    + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                    + "\n"
                )
        return nbest_hyps


    def post_process(
        self,
        i: int,
        ilens: torch.Tensor,
        maxlenratio: float,
        running_hyps: BatchHypothesis,
        ended_hyps: List[Hypothesis],
        stop_search: List[bool]
    ) -> BatchHypothesis:
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            ilens (torch.Tensor): The maximum output lengths of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (BatchHypothesis): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.
            stop_search (List[bool]): The ended flag in beam search.

        Returns:
            BatchHypothesis: The new running hypotheses.

        """
        n_batch = running_hyps.yseq.shape[0]
        batch_size = len(ilens)
        logging.debug(f"the number of running hypothes: {n_batch}")
        if self.token_list is not None:
            logging.debug(
                "best hypo: "
                + "".join(
                    [
                        self.token_list[x]
                        for x in running_hyps.yseq[0, 1 : running_hyps.length[0]]
                    ]
                )
            )
        # add eos in the final loop to avoid that there are no ended hyps
        maxlen = max(1, int(maxlenratio * max(ilens)))
        for utt_i in range(batch_size):
            if stop_search[utt_i]:
                # reset lengths of ended hyps
                running_hyps.length[utt_i * self.beam_size:(utt_i + 1) * self.beam_size]=torch.full((self.beam_size, ), 0, device=running_hyps.length.device, dtype=torch.int64) 
                continue
            if i == ilens[utt_i] - 1:
                stop_search[utt_i] = True
                logging.info(f"adding <eos> in the last position in th loop for utterance: ilens[utt_i]")
                # add ended hypotheses to a final list
                for beam_j in range(self.beam_size):
                    ended_hyps[utt_i].append(
                        Hypothesis(
                            yseq=self.append_token(running_hyps.yseq[utt_i * self.beam_size + beam_j], self.eos),
                            score=running_hyps.score[utt_i * self.beam_size + beam_j],
                            scores={k: v[utt_i * self.beam_size + beam_j] for k, v in running_hyps.scores.items()},
                            states={k: self.scorers[k].select_state(v, utt_i * self.beam_size + beam_j) for k, v in running_hyps.states.items()},
                        )
                    ) 
                    running_hyps.length[utt_i * self.beam_size + beam_j] = 0
                
        is_eos = (
            running_hyps.yseq[torch.arange(n_batch), running_hyps.length - 1]
            == self.eos
        )
        # mask stop search hyps
        end_mask = (running_hyps.length != 0).to(is_eos.device)
        is_eos = is_eos & end_mask
        ended_ids = torch.nonzero(is_eos).view(-1)
        for b in ended_ids:
            hyp = self._select(running_hyps, b)
            ended_hyps[b // self.beam_size].append(hyp)
        running_hyps = self._batch_reset(running_hyps, ended_ids)
        for utt_i in range(batch_size):
            if stop_search[utt_i]:
                continue
            stop_search[utt_i] = (running_hyps.length[utt_i * self.beam_size:(utt_i + 1)*self.beam_size].sum() == 0)
        return running_hyps
