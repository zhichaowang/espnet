#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parameter initialization for transducer RNN/Transformer parts."""

from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one

from espnet.nets.pytorch_backend.transformer.initializer import initialize


def initializer(model, args):
    """Initialize transducer model.

    Args:
        model (torch.nn.Module): transducer instance
        args (Namespace): argument Namespace containing options

    """
    if "custom" not in args.dtype:
        if "custom" in args.etype:
            initialize(model.encoder, args.transformer_init)
            lecun_normal_init_parameters(model.dec)
        else:
            lecun_normal_init_parameters(model)

        model.dec.embed.weight.data.normal_(0, 1)

        for i in range(model.dec.dlayers):
            set_forget_bias_to_one(getattr(model.dec.decoder[i], "bias_ih_l0"))
            set_forget_bias_to_one(getattr(model.dec.decoder[i], "bias_hh_l0"))
    else:
        if "custom" in args.etype:
            initialize(model, args.transformer_init)
        else:
            lecun_normal_init_parameters(model.enc)
            initialize(model.decoder, args.transformer_init)
