from collections import OrderedDict
from typing import List
from typing import Tuple

import torch
import torch.nn as nn

from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.enh.layers.u_net_layers import ConvLayer
from espnet2.enh.layers.u_net_layers import DownsamplingBlock
from espnet2.enh.layers.u_net_layers import UpsamplingBlock


class WaveUNet(AbsEnhancement):
    def __init__(
        self,
        num_input_chs: int = 2,
        num_output_chs: int = 2,
        num_spk: int = 2,
        input_feature_channels: int = 32,
        num_levels: int = 6,
        feature_growth: str = "double",
        kernel_size: int = 5,
        target_output_size: int = 32000,
        conv_type: str = "gn",
        res: str = "fixed",
        depth: int = 1,
        strides: int = 2,
        loss_type: str = "si_snr",
    ):
        """Main WaveUNet class.

        Args:
            num_input_chs: Number of input audio channels
            num_output_chs: Number of output audio channels
            num_spk: Number of speakers
            input_feature_channels: Number of feature channels in the first DS layer
            num_levels: Number of DS/US blocks
            feature_growth: How the features in each layer should grow.
                - "add": the initial number of features each time;
                - "double": multiply by 2.
            kernel_size: Filter width of kernels (an odd number)
            target_output_size: Output duration in samples
            conv_type: Type of convolution:
                - "normal": normal
                - "bn": BN-normalised
                - "gn": GN-normalised
            res: Resampling strategy:
                - "fixed": fixed sinc-based lowpass filtering
                - "learned": learned conv layer
            depth: Number of convs per block
            strides: Strides in Wave-U-Net
            loss_type: How to compute the loss

        Reference:
            Daniel Stoller, Sebastian Ewert, and Simon Dixon. Wave-U-Net:
            A multi-scale neural network for end-to-end audio source separation.
            In ISMIR, 2018.
        Based on https://github.com/f90/Wave-U-Net-Pytorch
        """
        super(WaveUNet, self).__init__()

        if feature_growth == "add":
            num_channels = [
                input_feature_channels * i for i in range(1, num_levels + 1)
            ]
        elif feature_growth == "double":
            num_channels = [
                input_feature_channels * 2 ** i for i in range(0, num_levels)
            ]
        else:
            raise ValueError("Unsupported feature_growth: {}".format(feature_growth))

        self.num_levels = num_levels
        self.strides = strides
        self.kernel_size = kernel_size

        self.num_input_chs = num_input_chs
        self.num_output_chs = num_output_chs
        self.num_spk = num_spk
        self.depth = depth
        self.loss_type = loss_type

        # Only odd filter kernels allowed
        assert kernel_size % 2 == 1, kernel_size
        assert conv_type in ("normal", "bn", "gn"), conv_type
        assert res in ("fixed", "learned"), res

        module = nn.Module()
        module.downsampling_blocks = nn.ModuleList()
        module.upsampling_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            in_ch = num_input_chs if i == 0 else num_channels[i]

            module.downsampling_blocks.append(
                DownsamplingBlock(
                    in_ch,
                    num_channels[i],
                    num_channels[i + 1],
                    kernel_size,
                    strides,
                    depth,
                    conv_type,
                    res,
                )
            )

        for i in range(0, self.num_levels - 1):
            module.upsampling_blocks.append(
                UpsamplingBlock(
                    num_channels[-1 - i],
                    num_channels[-2 - i],
                    num_channels[-2 - i],
                    kernel_size,
                    strides,
                    depth,
                    conv_type,
                    res,
                )
            )

        module.bottlenecks = nn.ModuleList(
            [
                ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1, conv_type)
                for _ in range(depth)
            ]
        )

        # Output conv
        outputs = num_output_chs * num_spk
        module.output_conv = nn.Conv1d(num_channels[0], outputs, 1)

        self.waveunet = module

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print(
            "Using valid convolutions with {} inputs and {} outputs.".format(
                self.input_size, self.output_size
            )
        )
        print(
            "Please make sure to use the chunk-based iterator "
            "and set chunk_length to {}.".format(self.input_size)
        )

        assert (self.input_size - self.output_size) % 2 == 0
        self.shapes = {
            "output_start_frame": (self.input_size - self.output_size) // 2,
            "output_end_frame": (self.input_size - self.output_size) // 2
            + self.output_size,
            "output_frames": self.output_size,
            "input_frames": self.input_size,
        }

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles
        # so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunet
        try:
            curr_size = bottleneck
            for block in module.upsampling_blocks:
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for block in reversed(module.downsampling_blocks):
                curr_size = block.get_input_size(curr_size)

            assert output_size >= target_output_size
            return curr_size, output_size
        except AssertionError:
            return False

    def forward_module(self, x, module):
        """A forward pass through a single Wave-U-Net.

        Multiple Wave-U-Nets might be used, one for each source.

        Args:
            x: Input mixture (Batch, Channels, Nsample)
            module: Network module to be used for prediction
        Returns:
            Source estimates (Batch, self.num_output_chs * self.num_spk, Nsample)
        """
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in module.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])

        # OUTPUT CONV
        out = module.output_conv(out)
        if not self.training:
            # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out

    def forward(self, x, ilens=None):
        # (Batch, Channel, Nsample)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        else:
            x = x.transpose(-1, -2)
        curr_input_size = x.shape[-1]
        # Users promise to feed the proper input themself,
        # to get the pre-calculated (NOT the originally desired) output size
        assert curr_input_size == self.input_size, (curr_input_size, self.input_size)

        out = self.forward_module(x, self.waveunet)

        out_lst = []
        for idx in range(self.num_spk):
            # List(Batch, output_frames, output_channels)
            out_lst.append(
                out[:, idx * self.num_output_chs : (idx + 1) * self.num_output_chs]
                .transpose(-1, -2)
                .squeeze(-1)
            )
        masks = OrderedDict()
        return out_lst, ilens, masks

    def forward_rawwav(self, x, ilens=None):
        return self.forward(x, ilens=ilens)

    def process_targets(
        self, input: torch.Tensor, target: List[torch.Tensor], ilens: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Process the target signals to match the model output size.

        Args:
            input: Input signal (Batch, Nsample [, Channels])
            target: Target signals List(Batch, Nsample [, Channels])
            ilens: Input/Target signal length (Batch,)
        Returns:
            target: Processed target signals List(Batch, Nsample2 [, Channels])
            olens: processed target signal length (Batch,)
        """
        # Crops target audio to the output shape required by self.shapes
        target = [
            tgt[
                :,
                self.shapes["output_start_frame"] : self.shapes["output_end_frame"],
            ]
            for tgt in target
        ]
        olens = torch.LongTensor([self.shapes["output_frames"] for _ in target])
        return target, olens
