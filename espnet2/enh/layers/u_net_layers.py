# modified from https://github.com/f90/Wave-U-Net-Pytorch
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


class UpsamplingBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_shortcut,
        n_outputs,
        kernel_size,
        stride,
        depth,
        conv_type,
        res,
    ):
        super(UpsamplingBlock, self).__init__()
        assert stride > 1

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(
                n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True
            )

        self.pre_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = centre_crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(
                torch.cat([combined, centre_crop(upsampled, combined)], dim=1)
            )
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size


class DownsamplingBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_shortcut,
        n_outputs,
        kernel_size,
        stride,
        depth,
        conv_type,
        res,
    ):
        super(DownsamplingBlock, self).__init__()
        assert stride > 1

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

        self.post_shortcut_convs = nn.ModuleList(
            [ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)]
            + [
                ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type)
                for _ in range(depth - 1)
            ]
        )

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(
                n_outputs, 15, stride
            )  # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(
                n_outputs, n_outputs, kernel_size, stride, conv_type
            )

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size


def centre_crop(x, target):
    """Center-crop 3-dim. input tensor along last axis
    so it fits the target tensor shape.

    Args:
        x: Input tensor
        target: Shape of this tensor will be used as target shape
    Returns:
        Cropped input tensor
    """  # noqa: H405
    if x is None:
        return None
    if target is None:
        return x

    target_shape = target.shape
    diff = x.shape[-1] - target_shape[-1]
    assert diff % 2 == 0
    crop = diff // 2

    if crop == 0:
        return x
    if crop < 0:
        raise ArithmeticError

    return x[:, :, crop:-crop].contiguous()


class Resample1d(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride,
        transpose=False,
        padding="reflect",
        trainable=False,
    ):
        """Creates a resampling layer for time series data (using 1D convolution).

        Args:
            channels (int): Number of features C at each time-step
            kernel_size (int): Width of sinc-based lowpass-filter
                        (>= 15 recommended for good filtering performance)
            stride (int): Resampling factor
            transpose (bool): False for downsampling, true for upsampling
            padding (str): Either "reflect" to pad or "valid" to not pad
            trainable (bool): Optionally activate this to train the lowpass-filter,
                              starting from the sinc initialisation
        """
        super(Resample1d, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels

        cutoff = 0.5 / stride

        assert kernel_size > 2
        assert (kernel_size - 1) % 2 == 0
        assert padding == "reflect" or padding == "valid"

        filter = build_sinc_filter(kernel_size, cutoff)

        self.filter = torch.nn.Parameter(
            torch.from_numpy(
                np.repeat(np.reshape(filter, [1, 1, kernel_size]), channels, axis=0)
            ),
            requires_grad=trainable,
        )

    def forward(self, x):
        # input shape: (N, C, W)
        # Pad here if not using transposed conv
        input_size = x.shape[2]
        if self.padding != "valid":
            num_pad = (self.kernel_size - 1) // 2
            out = F.pad(x, (num_pad, num_pad), mode=self.padding)
        else:
            out = x

        # Lowpass filter (+ 0 insertion if transposed)
        if self.transpose:
            expected_steps = (input_size - 1) * self.stride + 1
            if self.padding == "valid":
                expected_steps = expected_steps - self.kernel_size + 1

            out = F.conv_transpose1d(
                out, self.filter, stride=self.stride, padding=0, groups=self.channels
            )
            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert diff_steps % 2 == 0
                out = out[:, :, diff_steps // 2 : -diff_steps // 2]
        else:
            assert input_size % self.stride == 1
            out = F.conv1d(
                out, self.filter, stride=self.stride, padding=0, groups=self.channels
            )

        return out

    def get_output_size(self, input_size):
        """Returns the output dimensionality (number of timesteps)
        for a given input size.

        Args:
            input_size: Number of input time steps
                        (Scalar, each feature is one-dimensional)
        Returns:
            output_size (scalar)
        """  # noqa: H405
        assert input_size > 1
        if self.transpose:
            if self.padding == "valid":
                return ((input_size - 1) * self.stride + 1) - self.kernel_size + 1
            else:
                return (input_size - 1) * self.stride + 1
        else:
            assert input_size % self.stride == 1  # Want to take first and last sample
            if self.padding == "valid":
                return input_size - self.kernel_size + 1
            else:
                return input_size

    def get_input_size(self, output_size):
        """Returns the input dimensionality (number of timesteps)
        for a given output size.

        Args:
            input_size: Number of input time steps
                        (Scalar, each feature is one-dimensional)
        Returns:
            output_size (scalar)
        """  # noqa: H405

        # Strided conv/decimation
        if not self.transpose:
            curr_size = (
                output_size - 1
            ) * self.stride + 1  # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        if self.padding == "valid":
            curr_size = curr_size + self.kernel_size - 1  # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert (
                curr_size - 1
            ) % self.stride == 0  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert curr_size > 0
        return curr_size


def build_sinc_filter(kernel_size, cutoff):
    # FOLLOWING https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf (# noqa: E501)
    # Sinc lowpass filter
    # Build sinc kernel
    assert kernel_size % 2 == 1
    M = kernel_size - 1
    filter = np.zeros(kernel_size, dtype=np.float32)
    for i in range(kernel_size):
        if i == M // 2:
            filter[i] = 2 * np.pi * cutoff
        else:
            filter[i] = (np.sin(2 * np.pi * cutoff * (i - M // 2)) / (i - M // 2)) * (
                0.42 - 0.5 * np.cos((2 * np.pi * i) / M) + 0.08 * np.cos(4 * np.pi * M)
            )

    filter = filter / np.sum(filter)
    return filter


class ConvLayer(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, conv_type, transpose=False
    ):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type

        # How many channels should be normalised as one group if GroupNorm is activated
        # WARNING: Number of channels has to be divisible by this number!
        NORM_CHANNELS = 8

        if self.transpose:
            self.filter = nn.ConvTranspose1d(
                n_inputs, n_outputs, self.kernel_size, stride, padding=kernel_size - 1
            )
        else:
            self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        if conv_type == "gn":
            assert n_outputs % NORM_CHANNELS == 0
            self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)
        elif conv_type == "bn":
            self.norm = nn.BatchNorm1d(n_outputs, momentum=0.01)
        # Add you own types of variations here!

    def forward(self, x):
        # Apply the convolution
        if self.conv_type == "gn" or self.conv_type == "bn":
            out = F.relu(self.norm((self.filter(x))))
        elif self.conv_type == "normal":
            out = F.leaky_relu(self.filter(x))
        else:
            raise NotImplementedError("Unknown conv_type: {}".format(self.conv_type))
        return out

    def get_input_size(self, output_size):
        # Strided conv/decimation
        if not self.transpose:
            curr_size = (
                output_size - 1
            ) * self.stride + 1  # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        curr_size = curr_size + self.kernel_size - 1  # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert (
                curr_size - 1
            ) % self.stride == 0  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert curr_size > 0
        return curr_size

    def get_output_size(self, input_size):
        # Transposed
        if self.transpose:
            assert input_size > 1
            curr_size = (
                input_size - 1
            ) * self.stride + 1  # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = input_size

        # Conv
        curr_size = curr_size - self.kernel_size + 1  # o = i + p - k + 1
        assert curr_size > 0

        # Strided conv/decimation
        if not self.transpose:
            assert (
                curr_size - 1
            ) % self.stride == 0  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1

        return curr_size
