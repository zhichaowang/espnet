from typing import Optional

import torch
import torch.nn.functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.enh.espnet_model import EPS
from espnet2.enh.layers.beamformer import filter_minimum_gain_like
from espnet2.enh.layers.beamformer import get_adjacent
from espnet2.enh.layers.beamformer import get_mfmvdr_vector
from espnet2.enh.layers.beamformer import tik_reg
from espnet2.enh.layers.beamformer import vector_to_Hermitian
from espnet2.enh.layers.mask_estimator import TCNEstimator
from espnet2.layers.stft import Stft


class DeepMFMVDRNet(AbsEnhancement):
    """Single-channel Multi-Frame MVDR beamformer.

    Reference: https://arxiv.org/abs/2011.10345
    """

    def __init__(
        self,
        num_spk: int = 1,
        normalize_input: bool = False,
        mask_type: str = "IPM^2",
        loss_type: str = "si_snr",
        # STFT options
        n_fft: int = 128,
        win_length: int = None,
        hop_length: int = 32,
        center: bool = True,
        window: Optional[str] = "hann",
        normalized: bool = False,
        onesided: bool = True,
        # Beamformer options
        filter_length: int = 5,
        reg: float = 1e-3,
        minimum_gain: float = -17.0,
        minimum_gain_k: float = 10.0,
        hidden_dim: int = 128,
        layer: int = 4,
        stack: int = 2,
        kernel: int = 3,
        use_noise_mask: bool = True,
        beamformer_type: str = "mvdr_ifc",
    ):
        super(DeepMFMVDRNet, self).__init__()

        self.normalize_input = normalize_input
        self.mask_type = mask_type
        self.loss_type = loss_type
        if loss_type not in ("si_snr", "spectrum", "spectrum_log", "magnitude"):
            raise ValueError("Unsupported loss type: %s" % loss_type)

        self.num_spk = num_spk
        self.num_bin = n_fft // 2 + 1

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            window=window,
            normalized=normalized,
            onesided=onesided,
        )

        self.filter_length = filter_length
        output_size = filter_length ** 2 * self.num_bin
        self.reg = reg
        # converts dB to magnitude
        self.minimum_gain = 10 ** (minimum_gain / 20)
        self.minimum_gain_k = minimum_gain_k
        self.use_noise_mask = use_noise_mask
        self.beamformer_type = beamformer_type

        self.Phin_estimator = TCNEstimator(
            input_dim=2 * self.num_bin,
            output_dim=output_size,
            BN_dim=hidden_dim,
            hidden_dim=4 * hidden_dim,
            layer=layer,
            stack=stack,
            kernel=kernel,
        )

        self.Phiy_estimator = TCNEstimator(
            input_dim=2 * self.num_bin,
            output_dim=output_size,
            BN_dim=hidden_dim,
            hidden_dim=4 * hidden_dim,
            layer=layer,
            stack=stack,
            kernel=kernel,
        )

        self.xi_estimator = TCNEstimator(
            input_dim=self.num_bin,
            output_dim=self.num_bin,
            BN_dim=hidden_dim,
            hidden_dim=4 * hidden_dim,
            layer=layer,
            stack=stack,
            kernel=kernel,
        )

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, Nsample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            enhanced speech:
                torch.Tensor or List[torch.Tensor]
            output lengths
            predcited masks: None
        """
        batch_size = input.shape[0]
        # wave -> stft -> magnitude specturm
        input_spectrum, flens = self.stft(input, ilens)
        # (Batch, Freq, Frames)
        input_spectrum = ComplexTensor(
            input_spectrum[..., 0], input_spectrum[..., 1]
        ).transpose(-1, -2)
        if self.normalize_input:
            input_spectrum = input_spectrum / abs(input_spectrum.detach()).max()
        # Shape of input spectrum must be (Batch, Freq, Frames)
        assert input_spectrum.dim() == 3, input_spectrum.dim()

        # concatenate real and imaginary component over frequency dimension
        # (Batch, Freq * 2, Frames)
        stacked_spectrum = torch.cat([input_spectrum.real, input_spectrum.imag], dim=1)
        # get multi-frame vectors (Batch, Freq, Frames, filter_length)
        spec_adj = get_adjacent(input_spectrum, self.filter_length)

        # (Batch, Freq, Frames)
        log_magnitude = torch.log10(input_spectrum.abs())
        a_priori_snr = F.softplus(self.xi_estimator(log_magnitude))

        # (Batch, Freq * filter_length ** 2, Frames)
        correlation_noise = self.Phin_estimator(stacked_spectrum)
        # -> (Batch, Freq, Frames, filter_length ** 2)
        correlation_noise = (
            correlation_noise.view(
                (batch_size, self.num_bin, self.filter_length ** 2, -1)
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        # (Batch, Freq * filter_length ** 2, Frames)
        correlation_noisy = self.Phiy_estimator(stacked_spectrum)
        # -> (Batch, Freq, Frames, filter_length ** 2)
        correlation_noisy = (
            correlation_noisy.view(
                (batch_size, self.num_bin, self.filter_length ** 2, -1)
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        # assemble Hermitian matrices
        # (Batch, Freq, Frames, filter_length, filter_length)
        correlation_noise = vector_to_Hermitian(correlation_noise)
        correlation_noisy = vector_to_Hermitian(correlation_noisy)

        # force matrices to be psd
        correlation_noise = FC.matmul(
            correlation_noise, correlation_noise.transpose(-1, -2).conj()
        )
        correlation_noisy = FC.matmul(
            correlation_noisy, correlation_noisy.transpose(-1, -2).conj()
        )

        # Tikhonov regularization (if desired)
        if self.reg != 0.0:
            correlation_noise = tik_reg(correlation_noise, reg=self.reg)
            correlation_noisy = tik_reg(correlation_noisy, reg=self.reg)

        # compute inter-frame correlation (IFC) vectors
        gamman = correlation_noise[..., -1] / (
            correlation_noise[..., -1, -1, None] + EPS
        )
        gammay = correlation_noisy[..., -1] / (
            correlation_noisy[..., -1, -1, None] + EPS
        )
        gammax = ((1 + a_priori_snr) / (a_priori_snr + EPS))[..., None] * gammay - (
            1 / (a_priori_snr + EPS)
        )[..., None] * gamman

        # compute MFMVDR filters
        # (Batch, Freq, Frames, filter_length)
        filters = get_mfmvdr_vector(gammax, correlation_noise)

        # speech estimate (Batch, Freq, Frames)
        enhanced, _ = filter_minimum_gain_like(
            self.minimum_gain, filters, spec_adj, k=self.minimum_gain_k
        )
        # -> (Batch, Frames, Freq)
        enhanced = enhanced.transpose(-1, -2)
        # Convert ComplexTensor to torch.Tensor (Batch, Frames, Freq, 2)
        if isinstance(enhanced, list):
            # multi-speaker output
            enhanced = [torch.stack([enh.real, enh.imag], dim=-1) for enh in enhanced]
        else:
            # single-speaker output
            enhanced = [torch.stack([enhanced.real, enhanced.imag], dim=-1)]

        return enhanced, flens, None

    def forward_rawwav(self, input: torch.Tensor, ilens: torch.Tensor):
        """Output with wavformes.

        Args:
            input (torch.Tensor): mixed speech [Batch, Nsample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            predcited speech wavs:
                torch.Tensor(Batch, Nsamples), or List[torch.Tensor(Batch, Nsamples)]
            output lengths
            predcited masks: None
        """
        # predict spectrum for each speaker
        predicted_spectrums, _, masks = self.forward(input, ilens)

        if predicted_spectrums is None:
            predicted_wavs = None
        elif isinstance(predicted_spectrums, list):
            predicted_wavs = [
                self.stft.inverse(ps, ilens)[0] for ps in predicted_spectrums
            ]

        return predicted_wavs, ilens, masks
