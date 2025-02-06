# This script was created based on the sdfstudio_dataparser.py file.



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

"""Data parser for friends dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from torchtyping import TensorType

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs, Semantics
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.images import BasicImages
from nerfstudio.utils.io import load_from_json

CONSOLE = Console()

def get_src_from_pairs(
    ref_idx, all_imgs, pairs_srcs, neighbors_num=None, neighbors_shuffle=False
) -> Dict[str, TensorType]:
    # src_idx[0] is ref img
    src_idx = pairs_srcs[ref_idx]
    # randomly sample neighbors
    if neighbors_num and neighbors_num > -1 and neighbors_num < len(src_idx) - 1:
        if neighbors_shuffle:
            perm_idx = torch.randperm(len(src_idx) - 1) + 1
            src_idx = torch.cat([src_idx[[0]], src_idx[perm_idx[:neighbors_num]]])
        else:
            src_idx = src_idx[: neighbors_num + 1]
    src_idx = src_idx.to(all_imgs.device)
    return {"src_imgs": all_imgs[src_idx], "src_idxs": src_idx}

def get_image(image_filename, alpha_color=None) -> TensorType["image_height", "image_width", "num_channels"]:
    """Returns a 3 channel image.

    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is in correct."
    image = torch.from_numpy(np_image.astype("float32") / 255.0)
    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
    return image


def get_semantics(image_idx: int, semantic_path, include_auxiliary_semantics=False, aux_path=None, include_semantic_group=False, group_path=None):
    """function to process semantic labels

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # semantics
    semantic = np.array(Image.open(semantic_path[image_idx]), dtype="uint8")
    semantic = torch.from_numpy(semantic).type(torch.uint8)
    results = {"semantics": semantic}
    
    # auxiliary semantics
    if include_auxiliary_semantics:
        # print("auxiliary semantics")
        # import cv2
        # aux_data = cv2.imread(aux_path[image_idx]) 
        aux_data = np.array(Image.open(aux_path[image_idx]), dtype="uint8")
        aux_semantic = torch.from_numpy(aux_data[:,:,0]).type(torch.uint8)
        num_first_votes = torch.from_numpy(aux_data[:,:,1]).type(torch.uint8)
        num_second_votes = torch.from_numpy(aux_data[:,:,2]).type(torch.uint8)
        # avoid the situation that votes of first and second class both are 0
        num_first_votes[num_second_votes==0] = 1
        total_votes = num_second_votes + num_first_votes
        results['second_semantics'] = aux_semantic
        results['num_first_votes'] = num_first_votes
        results['num_second_votes'] = num_second_votes
        results['prob_first_class'] = num_first_votes / total_votes
        results['prob_second_class'] = num_second_votes / total_votes
        
    
    # groups for patch loss
    if include_semantic_group:
        assert group_path[image_idx] is not None
        groups_raw = np.array(Image.open(group_path[image_idx]), dtype="uint8")
        groups_raw = torch.from_numpy(groups_raw).type(torch.uint8)
        # make sure groups are unique and start from 0
        groups = torch.zeros_like(groups_raw)
        for i, group_id in enumerate(groups_raw.unique()):
            groups[groups_raw == group_id] = i
        groups = groups.type(torch.uint8)

        results["semantic_groups"] = groups
    
    return results

def get_instances(image_idx: int, instance_path):
    """function to process instance labels

    Args:
        image_idx: specific image index to work with
        semantics: instance data
    """

    # instances
    instance = np.array(Image.open(instance_path[image_idx]), dtype="uint8")
    instance = torch.from_numpy(instance).type(torch.uint8)
    results = {"instances": instance}    
    
    return results


@dataclass
class IMLDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: IML)
    """target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
    include_semantics: bool = False
    """whether or not to load semantic labels"""
    include_auxiliary_semantics: bool = False
    """whether or not to load auxiliary semantic labels"""
    include_semantic_groups: bool = False
    """whether or not to load semantic (or instance) segmentation group. """
    include_instances: bool = False
    """whether or not to load instance labels"""
    downscale_factor: int = 1
    scene_scale: float = 2.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Friends axis-aligned bbox will be scaled to this value.
    """
    load_pairs: bool = False
    """whether to load pairs for multi-view consistency"""
    neighbors_num: Optional[int] = None
    neighbors_shuffle: Optional[bool] = False
    pairs_sorted_ascending: Optional[bool] = True
    """if src image pairs are sorted in ascending order by similarity i.e.
    the last element is the most similar to the first (ref)"""
    skip_every_for_val_split: int = 1
    """sub sampling validation images"""
    train_val_no_overlap: bool = False
    """remove selected / sampled validation images from training set"""
    auto_orient: bool = False


@dataclass
class IML(DataParser):
    """Nerfacto Panoptic Dataset"""

    config: IMLDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        # load meta data
        meta = load_from_json(self.config.data / "meta_data.json")

        indices = list(range(len(meta["frames"])))
        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]
        else:
            # if you use this option, training set should not contain any image in validation set
            if self.config.train_val_no_overlap:
                indices = [i for i in indices if i % self.config.skip_every_for_val_split != 0]

        image_filenames = []
        semantic_filenames = []
        aux_semantic_filenames = []
        instance_filenames = []
        group_filenames = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for i, frame in enumerate(meta["frames"]):
            if i not in indices:
                continue

            image_filename = self.config.data / frame["rgb_path"]

            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])

            # append data
            image_filenames.append(image_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)

            if self.config.include_semantics:
                assert meta["has_semantics"]
                # load semantic labels
                filepath = str(self.config.data / frame["label_path"])
                semantic_filenames.append(filepath)
                if self.config.include_semantic_groups:
                    assert "group_path" in frame.keys()
                    filepath = str(self.config.data / frame["group_path"])
                    group_filenames.append(filepath)
                else:
                    group_filenames.append(None)
                if self.config.include_auxiliary_semantics:
                    assert "aux_label_path" in frame.keys()
                    filepath = str(self.config.data / frame["aux_label_path"])
                    aux_semantic_filenames.append(filepath)
                else:
                    aux_semantic_filenames.append(None)
                    
            if self.config.include_instances:
                assert meta["has_instances"]
                # load instance labels
                filepath = str(self.config.data / frame["instance_path"])
                instance_filenames.append(filepath)

        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        if self.config.auto_orient:
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method="up",
                center_poses=False,
            )

        # scene box from meta data
        meta_scene_box = meta["scene_box"]
        aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox(
            aabb=aabb,
            near=meta_scene_box["near"],
            far=meta_scene_box["far"],
            radius=meta_scene_box["radius"],
            collider_type=meta_scene_box["collider_type"],
        )

        height, width = meta["height"], meta["width"]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO supports downsample
        # cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        
        additional_inputs_dict = {}
        
        # load pair information
        pairs_path = self.config.data / "pairs.txt"
        if pairs_path.exists() and split == "train" and self.config.load_pairs:
            with open(pairs_path, "r") as f:
                pairs = f.readlines()
            split_ext = lambda x: x.split(".")[0]
            pairs_srcs = []
            for sources_line in pairs:
                sources_array = [int(split_ext(img_name)) for img_name in sources_line.split(" ")]
                if self.config.pairs_sorted_ascending:
                    # invert (flip) the source elements s.t. the most similar source is in index 1 (index 0 is reference)
                    sources_array = [sources_array[0]] + sources_array[:1:-1]
                pairs_srcs.append(sources_array)
            pairs_srcs = torch.tensor(pairs_srcs)
            all_imgs = torch.stack(
                [get_image(image_filename) for image_filename in sorted(image_filenames)], axis=0
            ).cuda()

            additional_inputs_dict["pairs"] = {
                "func": get_src_from_pairs,
                "kwargs": {
                    "all_imgs": all_imgs,
                    "pairs_srcs": pairs_srcs,
                    "neighbors_num": self.config.neighbors_num,
                    "neighbors_shuffle": self.config.neighbors_shuffle,
                },
            }

        # additionally load semantic labels in same format as friends
        if self.config.include_semantics:
            classes = [x['name'] for x in meta["semantic_classes"]]
            colors = [x['color'] for x in meta["semantic_classes"]]
            histogram = meta["semantic_class_histogram"]
            colors = torch.tensor(colors, dtype=torch.float32) / 255.0
            semantics = Semantics(filenames=semantic_filenames, classes=classes, colors=colors, histogram=histogram)

            additional_inputs_dict["semantics"] = {
                "func": get_semantics,
                "kwargs": {
                    "semantic_path": semantic_filenames, 
                    "include_auxiliary_semantics": self.config.include_auxiliary_semantics,
                    "aux_path": aux_semantic_filenames,
                    "include_semantic_group": self.config.include_semantic_groups,
                    "group_path": group_filenames
                },
            }
            
        # additionally load instance labels
        if self.config.include_instances:
            additional_inputs_dict["instance"] = {
                "func": get_instances,
                "kwargs": {
                    "instance_path": instance_filenames, 
                },
            }
            
        metadata_dic = {}
        if self.config.include_semantics:
            metadata_dic["semantics"] = semantics
        if self.config.include_instances:
            metadata_dic["num_instances"] = meta["num_instances"]

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs=additional_inputs_dict,
            metadata=metadata_dic,
        )
        return dataparser_outputs
