# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio, Accuracy, JaccardIndex
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
    SemanticRenderer,
    InstanceRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color


@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "white", "black"] = "last_sample"
    """Whether to randomize the background color."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 1024
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int] = (256, 96)
    """Number of samples per ray for the proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed noramls."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    instance_loss_mult: float = 0.0
    """instance loss multiplier"""
    semantic_loss_mult: float = 0.0
    """semantic loss multiplier"""
    semantic_num_layers: int = 4
    """Number of layers for semantic head."""
    semantic_layer_width: int = 256
    """Number of hidden dimension of semantic head."""
    instance_num_layers: int = 4
    """Number of layers for instance head."""
    instance_layer_width: int = 256
    """Number of hidden dimension of instance head."""
    semantic_ignore_label: int = -1
    """semantic index in label for ignore mask"""
    semantic_patch_loss_mult: float = 0.0
    """semantic patch consistency loss as in panoptic lifting"""
    semantic_patch_loss_min_step: int = 0
    """semantic patch consistency loss as in panoptic lifting"""


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def __init__(self, config, metadata: Dict, **kwargs) -> None:
        if metadata and metadata.keys():
            if "semantics" in metadata.keys():
                assert isinstance(metadata["semantics"], Semantics)
                self.semantics = metadata["semantics"]
            if "num_instances" in metadata.keys():
                self.num_instances = metadata["num_instances"]
            else:
                self.num_instances = 188
            
        super().__init__(config=config, metadata=metadata, **kwargs)
        
    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.scene_contraction = SceneContraction(order=float("inf"))

        #semantics
        if hasattr(self, 'semantics'):
            hist = np.array(self.semantics.histogram)
            # check which classes occur (plus 0 as ignore label), we only use these in the model
            self.semantics_model2output = np.argwhere(hist > 0).squeeze()
            if not self.config.semantic_ignore_label in self.semantics_model2output:
                self.semantics_model2output = torch.cat(
                    [torch.tensor([self.config.semantic_ignore_label]), torch.tensor(self.semantics_model2output)])
            else:
                self.semantics_model2output = torch.tensor(self.semantics_model2output)
            self.semantics_output2model = torch.zeros(hist.shape[0], dtype=torch.long)
            self.semantics_output2model[self.semantics_model2output] = torch.arange(
                self.semantics_model2output.shape[0], dtype=torch.long
            )
            self.semantics_numclasses = self.semantics_model2output.shape[0]
            hist = hist[self.semantics_model2output]
            hist = torch.tensor(hist / hist.sum()).float()
            ignore_label = self.semantics_output2model[self.config.semantic_ignore_label]
            self.semantic_loss = torch.nn.CrossEntropyLoss(
                weight=(1.0 / (0.1 + hist)), reduction="mean", ignore_index=ignore_label)
            self.aux_semantic_loss = torch.nn.CrossEntropyLoss(
                weight=(1.0 / (0.1 + hist)), reduction="none", ignore_index=ignore_label)
            self.groups_loss = torch.nn.CrossEntropyLoss(reduction="mean")
            
        # instances
        self.instance_loss = torch.nn.CrossEntropyLoss(
                weight=None, reduction="mean", ignore_index=ignore_label)
        
        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_instances=self.config.instance_loss_mult > 0,
            use_semantics=self.config.semantic_loss_mult > 0,
            num_semantic_classes=self.semantics_numclasses if hasattr(self, "semantics") else None,
            semantic_num_layers = self.config.semantic_num_layers,
            semantic_layer_width = self.config.semantic_layer_width,
            instance_num_layers = self.config.instance_num_layers,
            instance_layer_width = self.config.instance_layer_width,
            num_instances=self.num_instances,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=self.scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=self.scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()
        self.renderer_semantic = SemanticRenderer()
        self.renderer_instance = InstanceRenderer()
        
        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        if hasattr(self, 'semantics'):
            self.sem_accuracy = Accuracy(
                task='multiclass', 
                num_classes=self.semantics_numclasses, 
                ignore_index=self.config.semantic_ignore_label
            )
            self.sem_miou = JaccardIndex(
                task='multiclass', 
                num_classes=self.semantics_numclasses,
                ignore_index=self.config.semantic_ignore_label
            )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "weights": weights,
        }

        if self.config.predict_normals:
            outputs["normals"] = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            outputs["pred_normals"] = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)

        if FieldHeadNames.SEMANTICS in field_outputs:
            semantics = self.renderer_semantic(semantics=field_outputs[FieldHeadNames.SEMANTICS],
                                               weights=weights.detach())
            outputs["semantics"] = semantics
            
        if FieldHeadNames.INSTANCES in field_outputs:
            instances = self.renderer_instance(instances=field_outputs[FieldHeadNames.INSTANCES],
                                               weights=weights.detach())
            outputs["instances"] = instances
            
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if True or self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        if not self.training:
            outputs.update({
                "ray_points": self.scene_contraction(
                    ray_samples.frustums.get_start_positions()
                ),  # used for creating visiblity mask
            })
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None, step=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            
            
            # instance loss
            if "instances" in batch and self.config.instance_loss_mult > 0.0:
                instances_label = batch["instances"].long().to(self.device)
                instances_pred = outputs["instances"]                
                loss_dict["instance_loss"] = (
                        self.instance_loss(instances_pred, instances_label) * self.config.instance_loss_mult
                    )
                    
            # semantic loss
            if "semantics" in batch and self.config.semantic_loss_mult > 0.0:
                label = self.semantics_output2model[batch["semantics"].long()].to(self.device)
                second_label = self.semantics_output2model[batch["second_semantics"].long()].to(self.device)
                semantics_pred = outputs["semantics"]
                if "second_semantics" in batch:
                    loss_dict["semantic_loss"] = (
                        (batch['prob_second_class'] * self.aux_semantic_loss(semantics_pred, second_label) + 
                            batch['prob_first_class'] * self.aux_semantic_loss(semantics_pred, label) ).mean() * self.config.semantic_loss_mult
                    )
                else:
                    loss_dict["semantic_loss"] = (
                        self.semantic_loss(semantics_pred, label) * self.config.semantic_loss_mult
                    )
         
                reached_min_step_for_patch_loss = True
                if self.config.semantic_patch_loss_min_step > 0:
                    assert step is not None
                    if step < self.config.semantic_patch_loss_min_step:
                        reached_min_step_for_patch_loss = False
                if self.config.semantic_patch_loss_mult > 0.0 and reached_min_step_for_patch_loss:
                    # patch consistency as in panoptic lifting
                    groups = batch["semantic_groups"].to(self.device)[:, None].long()
                    target_means = torch.zeros((255, semantics_pred.shape[-1]), device=self.device)
                    target_means = target_means.scatter_reduce(
                        dim=0,
                        index=torch.tile(groups, (1, self.semantics_numclasses)),
                        src=semantics_pred,
                        reduce="mean",
                        include_self=False,
                    )
                    target = target_means.argmax(-1)[groups[:, 0]]

                    # ramp up loss multiplier
                    loss_mult = self.config.semantic_patch_loss_mult * (step - self.config.semantic_patch_loss_min_step) / 5000.0
                    loss_dict["semantic_patch_loss"] = (
                        self.groups_loss(semantics_pred, target) * loss_mult
                    )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        # normals to RGB for visualization. TODO: use a colormap
        if "normals" in outputs:
            images_dict["normals"] = (outputs["normals"] + 1.0) / 2.0
        if "pred_normals" in outputs:
            images_dict["pred_normals"] = (outputs["pred_normals"] + 1.0) / 2.0

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        if "semantics" in batch:
            label = self.semantics_output2model[batch["semantics"].long()].to(self.device)
            pred = torch.argmax(outputs["semantics"], dim=-1)
            colorized_pred = self.semantics.colors[pred]
            colorized_label = self.semantics.colors[label.long()]
            images_dict["semantics"] = torch.cat([colorized_label, colorized_pred], dim=1)
            metrics_dict["semantics_acc"] = self.sem_accuracy(pred, label)
            metrics_dict["semantics_miou"] = self.sem_miou(pred, label)
            
        if "instances" in batch:
            label = batch["instances"].long().to(self.device)
            pred = torch.argmax(outputs["instances"], dim=-1)
            colorized_pred = self.semantics.colors[pred]
            colorized_label = self.semantics.colors[label.long()]
            images_dict["instances"] = torch.cat([colorized_label, colorized_pred], dim=1)
            # metrics_dict["semantics_acc"] = self.sem_accuracy(pred, label)
            # metrics_dict["semantics_miou"] = self.sem_miou(pred, label)
            
        return metrics_dict, images_dict
