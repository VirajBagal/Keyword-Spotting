import torch
from torch import Tensor
from torchaudio import functional as F

class _AxisMasking(torch.nn.Module):
    r"""Apply masking to a spectrogram.

    Args:
        mask_param (int): Maximum possible length of the mask.
        axis (int): What dimension the mask is applied on.
        iid_masks (bool): Applies iid masks to each of the examples in the batch dimension.
            This option is applicable only when the input tensor is 4D.
    """
    __constants__ = ['mask_param', 'axis', 'iid_masks']

    def __init__(self, mask_param: int, axis: int, iid_masks: bool, prob = 0.5) -> None:

        super(_AxisMasking, self).__init__()
        self.mask_param = mask_param
        self.axis = axis
        self.iid_masks = iid_masks
        self.probability = prob

    def forward(self, specgram: Tensor, mask_value: float = 0.) -> Tensor:
        r"""
        Args:
            specgram (Tensor): Tensor of dimension (..., freq, time).
            mask_value (float): Value to assign to the masked columns.

        Returns:
            Tensor: Masked spectrogram of dimensions (..., freq, time).
        """
        # if iid_masks flag marked and specgram has a batch dimension

        if torch.rand(1) < self.probability:
            if self.iid_masks and specgram.dim() == 4:
                return F.mask_along_axis_iid(specgram, self.mask_param, mask_value, self.axis + 1)
            else:
                return F.mask_along_axis(specgram, self.mask_param, mask_value, self.axis)
        else:
            return specgram


class FrequencyMasking(_AxisMasking):
    r"""Apply masking to a spectrogram in the frequency domain.

    Args:
        freq_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, freq_mask_param).
        iid_masks (bool, optional): whether to apply different masks to each
            example/channel in the batch. (Default: ``False``)
            This option is applicable only when the input tensor is 4D.
    """

    def __init__(self, freq_mask_param: int, iid_masks: bool = False, prob: float = 0.5) -> None:
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks, prob)


class TimeMasking(_AxisMasking):
    r"""Apply masking to a spectrogram in the time domain.

    Args:
        time_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, time_mask_param).
        iid_masks (bool, optional): whether to apply different masks to each
            example/channel in the batch. (Default: ``False``)
            This option is applicable only when the input tensor is 4D.
    """

    def __init__(self, time_mask_param: int, iid_masks: bool = False, prob: float = 0.5) -> None:
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks, prob)