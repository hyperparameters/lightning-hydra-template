import torch
from torch import Tensor
from typing import Optional


def approx_l1_loss(pred, true, epsilon=10e-6):
    loss = torch.sqrt((pred - true) ** 2 + epsilon**2)
    return loss


class AlphaLoss(torch.nn.Module):
    def forward(self, input: Tensor, target: Tensor, mask: Tensor = None) -> Tensor:
        assert input.shape == target.shape, "input and target must be of same size"
        if mask is not None:
            assert (
                input.shape[2:] == mask.shape[2:]
            ), "mask must be of same size as input and target"
            loss = approx_l1_loss(input * mask, target * mask)
            loss = loss.sum() / mask.sum()
        else:
            loss = approx_l1_loss(input, target)
            loss = loss.mean()
        return loss


def composition_loss(pred_alpha, true_alpha, true_fg, true_bg=None, epsilon=10e-6):
    true_composition = true_fg * true_alpha
    pred_composition = true_fg * pred_alpha

    if true_bg is not None:
        true_composition += true_bg * (1 - true_alpha)
        pred_composition += true_bg * (1 - pred_alpha)

    loss = approx_l1_loss(pred_composition, true_composition, epsilon)
    loss = loss.mean()
    return loss


class CompositionLoss(torch.nn.Module):
    def forward(
        self,
        input: Tensor,
        target: Tensor,
        true_fg: Tensor,
        true_bg: Optional[Tensor] = None,
    ) -> Tensor:
        loss = composition_loss(input, target, true_fg, true_bg)
        return loss
