# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection of Losses.
"""
from enum import Enum
from typing import Dict, Literal, Optional, Tuple, cast

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.math import masked_reduction, normalized_depth_scale_and_shift

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

LOSSES = {"L1": L1Loss, "MSE": MSELoss}

EPS = 1.0e-7

# Sigma scale factor from Urban Radiance Fields (Rematas et al., 2022)
URF_SIGMA_SCALE_FACTOR = 3.0


class DepthLossType(Enum):
    """Types of depth losses for depth supervision."""

    DS_NERF = 1
    URF = 2
    SPARSENERF_RANKING = 3


FORCE_PSEUDODEPTH_LOSS = False
PSEUDODEPTH_COMPATIBLE_LOSSES = (DepthLossType.SPARSENERF_RANKING,)


def outer(
    t0_starts: Float[Tensor, "*batch num_samples_0"],
    t0_ends: Float[Tensor, "*batch num_samples_0"],
    t1_starts: Float[Tensor, "*batch num_samples_1"],
    t1_ends: Float[Tensor, "*batch num_samples_1"],
    y1: Float[Tensor, "*batch num_samples_1"],
) -> Float[Tensor, "*batch num_samples_0"]:
    """Faster version of

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64

    Args:
        t0_starts: start of the interval edges
        t0_ends: end of the interval edges
        t1_starts: start of the interval edges
        t1_ends: end of the interval edges
        y1: weights
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)

    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), side="right") - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), side="right")
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


def lossfun_outer(
    t: Float[Tensor, "*batch num_samples_1"],
    w: Float[Tensor, "*batch num_samples"],
    t_env: Float[Tensor, "*batch num_samples_1"],
    w_env: Float[Tensor, "*batch num_samples"],
):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80

    Args:
        t: interval edges
        w: weights
        t_env: interval edges of the upper bound enveloping histogram
        w_env: weights that should upper bound the inner (t,w) histogram
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


def ray_samples_to_sdist(ray_samples):
    """Convert ray samples to s space"""
    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends
    sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)  # (num_rays, num_samples + 1)
    return sdist


def interlevel_loss(weights_list, ray_samples_list) -> torch.Tensor:
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    assert len(ray_samples_list) > 0

    loss_interlevel = 0.0
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))

    assert isinstance(loss_interlevel, Tensor)
    return loss_interlevel


# Verified
def lossfun_distortion(t, w):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def distortion_loss(weights_list, ray_samples_list):
    """From mipnerf360"""
    c = ray_samples_to_sdist(ray_samples_list[-1])
    w = weights_list[-1][..., 0]
    loss = torch.mean(lossfun_distortion(c, w))
    return loss


def nerfstudio_distortion_loss(
    ray_samples: RaySamples,
    densities: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
    weights: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
) -> Float[Tensor, "*bs 1"]:
    """Ray based distortion loss proposed in MipNeRF-360. Returns distortion Loss.

    .. math::

        \\mathcal{L}(\\mathbf{s}, \\mathbf{w}) =\\iint\\limits_{-\\infty}^{\\,\\,\\,\\infty}
        \\mathbf{w}_\\mathbf{s}(u)\\mathbf{w}_\\mathbf{s}(v)|u - v|\\,d_{u}\\,d_{v}

    where :math:`\\mathbf{w}_\\mathbf{s}(u)=\\sum_i w_i \\mathbb{1}_{[\\mathbf{s}_i, \\mathbf{s}_{i+1})}(u)`
    is the weight at location :math:`u` between bin locations :math:`s_i` and :math:`s_{i+1}`.

    Args:
        ray_samples: Ray samples to compute loss over
        densities: Predicted sample densities
        weights: Predicted weights from densities and sample locations
    """
    if torch.is_tensor(densities):
        assert not torch.is_tensor(weights), "Cannot use both densities and weights"
        assert densities is not None
        # Compute the weight at each sample location
        weights = ray_samples.get_weights(densities)
    if torch.is_tensor(weights):
        assert not torch.is_tensor(densities), "Cannot use both densities and weights"
    assert weights is not None

    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends

    assert starts is not None and ends is not None, "Ray samples must have spacing starts and ends"
    midpoints = (starts + ends) / 2.0  # (..., num_samples, 1)

    loss = (
        weights * weights[..., None, :, 0] * torch.abs(midpoints - midpoints[..., None, :, 0])
    )  # (..., num_samples, num_samples)
    loss = torch.sum(loss, dim=(-1, -2))[..., None]  # (..., num_samples)
    loss = loss + 1 / 3.0 * torch.sum(weights**2 * (ends - starts), dim=-2)

    return loss


def orientation_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],
    viewdirs: Float[Tensor, "*bs 3"],
):
    """Orientation loss proposed in Ref-NeRF.
    Loss that encourages that all visible normals are facing towards the camera.
    """
    w = weights
    n = normals
    v = viewdirs * -1
    n_dot_v = (n * v[..., None, :]).sum(dim=-1)
    return (w[..., 0] * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)


def pred_normal_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],
    pred_normals: Float[Tensor, "*bs num_samples 3"],
):
    """Loss between normals calculated from density and normals from prediction network."""
    return (weights[..., 0] * (1.0 - torch.sum(normals * pred_normals, dim=-1))).sum(dim=-1)


def ds_nerf_depth_loss(
    weights: Float[Tensor, "*batch num_samples 1"],
    termination_depth: Float[Tensor, "*batch 1"],
    steps: Float[Tensor, "*batch num_samples 1"],
    lengths: Float[Tensor, "*batch num_samples 1"],
    sigma: Float[Tensor, "0"],
) -> Float[Tensor, "*batch 1"]:
    """Depth loss from Depth-supervised NeRF (Deng et al., 2022).

    Args:
        weights: Weights predicted for each sample.
        termination_depth: Ground truth depth of rays.
        steps: Sampling distances along rays.
        lengths: Distances between steps.
        sigma: Uncertainty around depth values.
    Returns:
        Depth loss scalar.
    """
    depth_mask = termination_depth > 0

    loss = -torch.log(weights + EPS) * torch.exp(-((steps - termination_depth[:, None]) ** 2) / (2 * sigma)) * lengths
    loss = loss.sum(-2) * depth_mask
    return torch.mean(loss)


def urban_radiance_field_depth_loss(
    weights: Float[Tensor, "*batch num_samples 1"],
    termination_depth: Float[Tensor, "*batch 1"],
    predicted_depth: Float[Tensor, "*batch 1"],
    steps: Float[Tensor, "*batch num_samples 1"],
    sigma: Float[Tensor, "0"],
) -> Float[Tensor, "*batch 1"]:
    """Lidar losses from Urban Radiance Fields (Rematas et al., 2022).

    Args:
        weights: Weights predicted for each sample.
        termination_depth: Ground truth depth of rays.
        predicted_depth: Depth prediction from the network.
        steps: Sampling distances along rays.
        sigma: Uncertainty around depth values.
    Returns:
        Depth loss scalar.
    """
    depth_mask = termination_depth > 0

    # Expected depth loss
    expected_depth_loss = (termination_depth - predicted_depth) ** 2

    # Line of sight losses
    target_distribution = torch.distributions.normal.Normal(0.0, sigma / URF_SIGMA_SCALE_FACTOR)
    termination_depth = termination_depth[:, None]
    line_of_sight_loss_near_mask = torch.logical_and(
        steps <= termination_depth + sigma, steps >= termination_depth - sigma
    )
    line_of_sight_loss_near = (weights - torch.exp(target_distribution.log_prob(steps - termination_depth))) ** 2
    line_of_sight_loss_near = (line_of_sight_loss_near_mask * line_of_sight_loss_near).sum(-2)
    line_of_sight_loss_empty_mask = steps < termination_depth - sigma
    line_of_sight_loss_empty = (line_of_sight_loss_empty_mask * weights**2).sum(-2)
    line_of_sight_loss = line_of_sight_loss_near + line_of_sight_loss_empty

    loss = (expected_depth_loss + line_of_sight_loss) * depth_mask
    return torch.mean(loss)


def depth_loss(
    weights: Float[Tensor, "*batch num_samples 1"],
    ray_samples: RaySamples,
    termination_depth: Float[Tensor, "*batch 1"],
    predicted_depth: Float[Tensor, "*batch 1"],
    sigma: Float[Tensor, "0"],
    directions_norm: Float[Tensor, "*batch 1"],
    is_euclidean: bool,
    depth_loss_type: DepthLossType,
) -> Float[Tensor, "0"]:
    """Implementation of depth losses.

    Args:
        weights: Weights predicted for each sample.
        ray_samples: Samples along rays corresponding to weights.
        termination_depth: Ground truth depth of rays.
        predicted_depth: Depth prediction from the network.
        sigma: Uncertainty around depth value.
        directions_norm: Norms of ray direction vectors in the camera frame.
        is_euclidean: Whether ground truth depths corresponds to normalized direction vectors.
        depth_loss_type: Type of depth loss to apply.

    Returns:
        Depth loss scalar.
    """
    if not is_euclidean:
        termination_depth = termination_depth * directions_norm
    steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

    if depth_loss_type == DepthLossType.DS_NERF:
        lengths = ray_samples.frustums.ends - ray_samples.frustums.starts
        return ds_nerf_depth_loss(weights, termination_depth, steps, lengths, sigma)

    if depth_loss_type == DepthLossType.URF:
        return urban_radiance_field_depth_loss(weights, termination_depth, predicted_depth, steps, sigma)

    raise NotImplementedError("Provided depth loss type not implemented.")


def monosdf_normal_loss(
    normal_pred: Float[Tensor, "num_samples 3"], normal_gt: Float[Tensor, "num_samples 3"]
) -> Float[Tensor, "0"]:
    """
    Normal consistency loss proposed in monosdf - https://niujinshuchong.github.io/monosdf/
    Enforces consistency between the volume rendered normal and the predicted monocular normal.
    With both angluar and L1 loss. Eq 14 https://arxiv.org/pdf/2206.00665.pdf
    Args:
        normal_pred: volume rendered normal
        normal_gt: monocular normal
    """
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
    cos = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
    return l1 + cos


class MiDaSMSELoss(nn.Module):
    """
    data term from MiDaS paper
    """

    def __init__(self, reduction_type: Literal["image", "batch"] = "batch"):
        super().__init__()

        self.reduction_type: Literal["image", "batch"] = reduction_type
        # reduction here is different from the image/batch-based reduction. This is either "mean" or "sum"
        self.mse_loss = MSELoss(reduction="none")

    def forward(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            mse loss based on reduction function
        """
        summed_mask = torch.sum(mask, (1, 2))
        image_loss = torch.sum(self.mse_loss(prediction, target) * mask, (1, 2))
        # multiply by 2 magic number?
        image_loss = masked_reduction(image_loss, 2 * summed_mask, self.reduction_type)

        return image_loss


# losses based on https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py
class GradientLoss(nn.Module):
    """
    multiscale, scale-invariant gradient matching term to the disparity space.
    This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
    More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
    """

    def __init__(self, scales: int = 4, reduction_type: Literal["image", "batch"] = "batch"):
        """
        Args:
            scales: number of scales to use
            reduction_type: either "batch" or "image"
        """
        super().__init__()
        self.reduction_type: Literal["image", "batch"] = reduction_type
        self.__scales = scales

    def forward(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            gradient loss based on reduction function
        """
        assert self.__scales >= 1
        total = 0.0

        for scale in range(self.__scales):
            step = pow(2, scale)

            grad_loss = self.gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
            )
            total += grad_loss

        assert isinstance(total, Tensor)
        return total

    def gradient_loss(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        multiscale, scale-invariant gradient matching term to the disparity space.
        This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
        More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            reduction: reduction function, either reduction_batch_based or reduction_image_based
        Returns:
            gradient loss based on reduction function
        """
        summed_mask = torch.sum(mask, (1, 2))
        diff = prediction - target
        diff = torch.mul(mask, diff)

        grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)

        grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
        mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)

        image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
        image_loss = masked_reduction(image_loss, summed_mask, self.reduction_type)

        return image_loss


class ScaleAndShiftInvariantLoss(nn.Module):
    """
    Scale and shift invariant loss as described in
    "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
    https://arxiv.org/pdf/1907.01341.pdf
    """

    def __init__(self, alpha: float = 0.5, scales: int = 4, reduction_type: Literal["image", "batch"] = "batch"):
        """
        Args:
            alpha: weight of the regularization term
            scales: number of scales to use
            reduction_type: either "batch" or "image"
        """
        super().__init__()
        self.__data_loss = MiDaSMSELoss(reduction_type=reduction_type)
        self.__regularization_loss = GradientLoss(scales=scales, reduction_type=reduction_type)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        Args:
            prediction: predicted depth map (unnormalized)
            target: ground truth depth map (normalized)
            mask: mask of valid pixels
        Returns:
            scale and shift invariant loss
        """
        scale, shift = normalized_depth_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        """
        scale and shift invariant prediction
        from https://arxiv.org/pdf/1907.01341.pdf equation 1
        """
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


# end copy


# copy from https://github.com/svip-lab/Indoor-SfMLearner/blob/0d682b7ce292484e5e3e2161fc9fc07e2f5ca8d1/layers.py#L218
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self, patch_size):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(patch_size, 1)
        self.mu_y_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_x_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_y_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_xy_pool = nn.AvgPool2d(patch_size, 1)

        self.refl = nn.ReflectionPad2d(patch_size // 2)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


# TODO test different losses
class NCC(nn.Module):
    """Layer to compute the normalization cross correlation (NCC) of patches"""

    def __init__(self, patch_size: int = 11, min_patch_variance: float = 0.01):
        super(NCC, self).__init__()
        self.patch_size = patch_size
        self.min_patch_variance = min_patch_variance

    def forward(self, x, y):
        # TODO if we use gray image we should do it right after loading the image to save computations
        # to gray image
        x = torch.mean(x, dim=1)
        y = torch.mean(y, dim=1)

        x_mean = torch.mean(x, dim=(1, 2), keepdim=True)
        y_mean = torch.mean(y, dim=(1, 2), keepdim=True)

        x_normalized = x - x_mean
        y_normalized = y - y_mean

        norm = torch.sum(x_normalized * y_normalized, dim=(1, 2))
        var = torch.square(x_normalized).sum(dim=(1, 2)) * torch.square(y_normalized).sum(dim=(1, 2))
        denom = torch.sqrt(var + 1e-6)

        ncc = norm / (denom + 1e-6)

        # ignore pathces with low variances
        not_valid = (torch.square(x_normalized).sum(dim=(1, 2)) < self.min_patch_variance) | (
            torch.square(y_normalized).sum(dim=(1, 2)) < self.min_patch_variance
        )
        ncc[not_valid] = 1.0

        score = 1 - ncc.clip(-1.0, 1.0)  # 0->2: smaller, better
        return score[:, None, None, None]


class MultiViewLoss(nn.Module):
    """compute multi-view consistency loss"""

    def __init__(self, patch_size: int = 11, topk: int = 4, min_patch_variance: float = 0.01):
        super(MultiViewLoss, self).__init__()
        self.patch_size = patch_size
        self.topk = topk
        self.min_patch_variance = min_patch_variance
        # TODO make metric configurable
        # self.ssim = SSIM(patch_size=patch_size)
        # self.ncc = NCC(patch_size=patch_size)
        self.ssim = NCC(patch_size=patch_size, min_patch_variance=min_patch_variance)

        self.iter = 0

    def forward(self, patches: torch.Tensor, valid: torch.Tensor):
        """take the mim

        Args:
            patches (torch.Tensor): _description_
            valid (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        num_imgs, num_rays, _, num_channels = patches.shape

        if num_rays <= 0:
            return torch.tensor(0.0).to(patches.device)

        ref_patches = (
            patches[:1, ...]
            .reshape(1, num_rays, self.patch_size, self.patch_size, num_channels)
            .expand(num_imgs - 1, num_rays, self.patch_size, self.patch_size, num_channels)
            .reshape(-1, self.patch_size, self.patch_size, num_channels)
            .permute(0, 3, 1, 2)
        )  # [N_src*N_rays, 3, patch_size, patch_size]
        src_patches = (
            patches[1:, ...]
            .reshape(num_imgs - 1, num_rays, self.patch_size, self.patch_size, num_channels)
            .reshape(-1, self.patch_size, self.patch_size, num_channels)
            .permute(0, 3, 1, 2)
        )  # [N_src*N_rays, 3, patch_size, patch_size]

        # apply same reshape to the valid mask
        src_patches_valid = (
            valid[1:, ...]
            .reshape(num_imgs - 1, num_rays, self.patch_size, self.patch_size, 1)
            .reshape(-1, self.patch_size, self.patch_size, 1)
            .permute(0, 3, 1, 2)
        )  # [N_src*N_rays, 1, patch_size, patch_size]

        ssim = self.ssim(ref_patches.detach(), src_patches)
        ssim = torch.mean(ssim, dim=(1, 2, 3))
        ssim = ssim.reshape(num_imgs - 1, num_rays)

        # ignore invalid patch by setting ssim error to very large value
        ssim_valid = (
            src_patches_valid.reshape(-1, self.patch_size * self.patch_size).all(dim=-1).reshape(num_imgs - 1, num_rays)
        )
        # we should mask the error after we select the topk value, otherwise we might select far way patches that happens to be inside the image
        # ssim[torch.logical_not(ssim_valid)] = 1.1  # max ssim_error is 1

        min_ssim, idx = torch.topk(ssim, k=self.topk, largest=False, dim=0, sorted=True)

        min_ssim_valid = ssim_valid[idx, torch.arange(num_rays)[None].expand_as(idx)]
        # TODO how to set this value for better visualization
        min_ssim[torch.logical_not(min_ssim_valid)] = 0.0  # max ssim_error is 1

        if False:

            # visualization of topK error computations

            import cv2
            import numpy as np

            vis_patch_num = num_rays
            K = min(100, vis_patch_num)

            image = (
                patches[:, :vis_patch_num, :, :]
                .reshape(-1, vis_patch_num, self.patch_size, self.patch_size, 3)
                .permute(1, 2, 0, 3, 4)
                .reshape(vis_patch_num * self.patch_size, -1, 3)
            )

            src_patches_reshaped = src_patches.reshape(
                num_imgs - 1, num_rays, 3, self.patch_size, self.patch_size
            ).permute(1, 0, 3, 4, 2)
            idx = idx.permute(1, 0)

            selected_patch = (
                src_patches_reshaped[torch.arange(num_rays)[:, None].expand(idx.shape), idx]
                .permute(0, 2, 1, 3, 4)
                .reshape(num_rays, self.patch_size, self.topk * self.patch_size, 3)[:vis_patch_num]
                .reshape(-1, self.topk * self.patch_size, 3)
            )

            # apply same reshape to the valid mask
            src_patches_valid_reshaped = src_patches_valid.reshape(
                num_imgs - 1, num_rays, 1, self.patch_size, self.patch_size
            ).permute(1, 0, 3, 4, 2)

            selected_patch_valid = (
                src_patches_valid_reshaped[torch.arange(num_rays)[:, None].expand(idx.shape), idx]
                .permute(0, 2, 1, 3, 4)
                .reshape(num_rays, self.patch_size, self.topk * self.patch_size, 1)[:vis_patch_num]
                .reshape(-1, self.topk * self.patch_size, 1)
            )
            # valid to image
            selected_patch_valid = selected_patch_valid.expand_as(selected_patch).float()
            # breakpoint()

            image = torch.cat([selected_patch_valid, selected_patch, image], dim=1)
            # select top rays with highest errors

            image = image.reshape(num_rays, self.patch_size, -1, 3)

            _, idx2 = torch.topk(
                torch.sum(min_ssim, dim=0) / (min_ssim_valid.float().sum(dim=0) + 1e-6),
                k=K,
                largest=True,
                dim=0,
                sorted=True,
            )

            image = image[idx2].reshape(K * self.patch_size, -1, 3)

            cv2.imwrite(f"vis/{self.iter}.png", (image.detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1])
            self.iter += 1
            if self.iter == 9:
                breakpoint()

        return torch.sum(min_ssim) / (min_ssim_valid.float().sum() + 1e-6)


# sensor depth loss, adapted from https://github.com/dazinovic/neural-rgbd-surface-reconstruction/blob/main/losses.py
class SensorDepthLoss(nn.Module):
    """Sensor Depth loss"""

    def __init__(self, truncation: float):
        super(SensorDepthLoss, self).__init__()
        self.truncation = truncation  #  0.05 * 0.3 5cm scaled

    def forward(self, batch, outputs):
        """take the mim

        Args:
            batch (Dict): inputs
            outputs (Dict): outputs data from surface model

        Returns:
            l1_loss: l1 loss
            freespace_loss: free space loss
            sdf_loss: sdf loss
        """
        depth_pred = outputs["depth"]
        depth_gt = batch["sensor_depth"].to(depth_pred.device)[..., None]
        valid_gt_mask = depth_gt > 0.0

        l1_loss = torch.sum(valid_gt_mask * torch.abs(depth_gt - depth_pred)) / (valid_gt_mask.sum() + 1e-6)

        # free space loss and sdf loss
        ray_samples = outputs["ray_samples"]
        filed_outputs = outputs["field_outputs"]
        pred_sdf = filed_outputs[FieldHeadNames.SDF][..., 0]
        directions_norm = outputs["directions_norm"]

        z_vals = ray_samples.frustums.starts[..., 0] / directions_norm

        truncation = self.truncation
        front_mask = valid_gt_mask & (z_vals < (depth_gt - truncation))
        back_mask = valid_gt_mask & (z_vals > (depth_gt + truncation))
        sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)

        num_fs_samples = front_mask.sum()
        num_sdf_samples = sdf_mask.sum()
        num_samples = num_fs_samples + num_sdf_samples + 1e-6
        fs_weight = 1.0 - num_fs_samples / num_samples
        sdf_weight = 1.0 - num_sdf_samples / num_samples

        free_space_loss = torch.mean((F.relu(truncation - pred_sdf) * front_mask) ** 2) * fs_weight

        sdf_loss = torch.mean(((z_vals + pred_sdf) - depth_gt) ** 2 * sdf_mask) * sdf_weight

        return l1_loss, free_space_loss, sdf_loss
def tv_loss(grids: Float[Tensor, "grids feature_dim row column"]) -> Float[Tensor, ""]:
    """
    https://github.com/apchenstu/TensoRF/blob/4ec894dc1341a2201fe13ae428631b58458f105d/utils.py#L139

    Args:
        grids: stacks of explicit feature grids (stacked at dim 0)
    Returns:
        average total variation loss for neighbor rows and columns.
    """
    number_of_grids = grids.shape[0]
    h_tv_count = grids[:, :, 1:, :].shape[1] * grids[:, :, 1:, :].shape[2] * grids[:, :, 1:, :].shape[3]
    w_tv_count = grids[:, :, :, 1:].shape[1] * grids[:, :, :, 1:].shape[2] * grids[:, :, :, 1:].shape[3]
    h_tv = torch.pow((grids[:, :, 1:, :] - grids[:, :, :-1, :]), 2).sum()
    w_tv = torch.pow((grids[:, :, :, 1:] - grids[:, :, :, :-1]), 2).sum()
    return 2 * (h_tv / h_tv_count + w_tv / w_tv_count) / number_of_grids


class _GradientScaler(torch.autograd.Function):  # typing: ignore
    """
    Scale gradients by a constant factor.
    """

    @staticmethod
    def forward(ctx, value, scaling):
        ctx.save_for_backward(scaling)
        return value, scaling

    @staticmethod
    def backward(ctx, output_grad, grad_scaling):
        (scaling,) = ctx.saved_tensors
        return output_grad * scaling, grad_scaling


def scale_gradients_by_distance_squared(
    field_outputs: Dict[FieldHeadNames, torch.Tensor],
    ray_samples: RaySamples,
) -> Dict[FieldHeadNames, torch.Tensor]:
    """
    Scale gradients by the ray distance to the pixel
    as suggested in `Radiance Field Gradient Scaling for Unbiased Near-Camera Training` paper

    Note: The scaling is applied on the interval of [0, 1] along the ray!

    Example:
        GradientLoss should be called right after obtaining the densities and colors from the field. ::
            >>> field_outputs = scale_gradient_by_distance_squared(field_outputs, ray_samples)
    """
    out = {}
    ray_dist = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
    scaling = torch.square(ray_dist).clamp(0, 1)
    for key, value in field_outputs.items():
        out[key], _ = cast(Tuple[Tensor, Tensor], _GradientScaler.apply(value, scaling))
    return out


def depth_ranking_loss(rendered_depth, gt_depth):
    """
    Depth ranking loss as described in the SparseNeRF paper
    Assumes that the layout of the batch comes from a PairPixelSampler, so that adjacent samples in the gt_depth
    and rendered_depth are from pixels with a radius of each other
    """
    m = 1e-4
    dpt_diff = gt_depth[::2, :] - gt_depth[1::2, :]
    out_diff = rendered_depth[::2, :] - rendered_depth[1::2, :] + m
    differing_signs = torch.sign(dpt_diff) != torch.sign(out_diff)
    return torch.nanmean((out_diff[differing_signs] * torch.sign(out_diff[differing_signs])))
