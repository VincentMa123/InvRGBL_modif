import os
import time
import random
from enum import IntEnum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
import logging
import kornia
import viser
import nerfview
from bvh import RayTracer
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .pbr import (
    rendering_equation,
    rendering_equation_lidar,
    compute_spotlight_contribution,
    compute_screen_space_ao,
    image_space_pbr,
)
from models.gaussians.basics import *
from utils.graphics_utils import sample_incident_rays
from datasets.base.pixel_source import get_rays
from models.losses import (
    normal_map_smooth_loss,
    neighborhood_smoothness_loss,
    region_consistency_loss_from_labels,
)

logger = logging.getLogger()



class GSModelType(IntEnum):
    Background = 0
    RigidNodes = 1
    SMPLNodes = 2
    DeformableNodes = 3

def lr_scheduler_fn(
    cfg: OmegaConf,
    lr_init: float
):
    if cfg.lr_final is None:
        lr_final = lr_init
    else:
        lr_final = cfg.lr_final

    def func(step):
        step = step - cfg.opt_after
        if step < 0:
            return 0.
        
        if step < cfg.warmup_steps:
            if cfg.ramp == "cosine":
                lr = cfg.lr_pre_warmup + (lr_init - cfg.lr_pre_warmup) * np.sin(
                    0.5 * np.pi * np.clip(step / cfg.warmup_steps, 0, 1)
                )
            else:
                lr = (
                    cfg.lr_pre_warmup
                    + (lr_init - cfg.lr_pre_warmup) * step / cfg.warmup_steps
                )
        else:
            t = np.clip(
                (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps), 0, 1
            )
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return lr  # divided by lr_init because the multiplier is with the initial learning rate

    return func

class BasicTrainer(nn.Module):
    def __init__(
        self,
        type: str = "basic",
        optim: OmegaConf = None,
        losses: OmegaConf = None,
        render: OmegaConf = None,
        res_schedule: OmegaConf = None,
        gaussian_optim_general_cfg: OmegaConf = None,
        gaussian_ctrl_general_cfg: OmegaConf = None,
        model_config: OmegaConf = None,
        num_train_images: int = 0,
        num_full_images: int = 0,
        test_set_indices: List[int] = None,
        scene_aabb: torch.Tensor = None,
        device=None,
        freeze_step: OmegaConf = None
    ):
        super().__init__()
        self._type = type
        self.optim_general = optim
        self.losses_dict = losses
        self.render_cfg = render
        self.res_schedule = res_schedule
        self.model_config = model_config
        self.num_iters = self.optim_general.get("num_iters", 30000)
        self.gaussian_optim_general_cfg = gaussian_optim_general_cfg
        self.gaussian_ctrl_general_cfg = gaussian_ctrl_general_cfg
        self.step = 0
        self.device = device

        self._visibility_tracings_list = {}
        self._incident_dirs_list = {}
        self._incident_areas_list = {}    
        self._sun_visibility_tracings_list = {}
        self.labels = {}
        
        self.freeze_step = freeze_step 
        self.freezed = False

        # dataset infos
        self.num_train_images = num_train_images
        self.num_full_images = num_full_images
        
        # init scene scale
        self._init_scene(scene_aabb=scene_aabb)
        
        # init models
        self.models = {}
        self.misc_classes_keys = [
            'Sky', 'Affine', 'CamPose', 'CamPosePerturb', 'EnvMap'
        ]
        self.gaussian_classes = {}
        self._init_models()
        
        # background color
        self.back_color = torch.zeros(3).to(self.device)
        # for evaluation
        self.cur_frame = torch.tensor(0, device=self.device)
        self.test_set_indices = test_set_indices # will be override
        # a simple viewer for background visualization
        self.viewer = None
        self.pbr = self.render_cfg.pbr
        
        # Add environment map for PBR if not already created by config
        if self.pbr and 'EnvMap' not in self.models:
            from models.modules_envmap import EnvironmentMap
            self.models['EnvMap'] = EnvironmentMap(h=32, w=64).to(self.device)
        
        self.pts_labels = None # will be overwritten in forward
        self.render_dynamic_mask = False
        
        # init losses fn
        self._init_losses()
        
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        self.step = 0
        self.sun_intensity = 10
        self.spotlights = None  # inference-time spotlights for relighting

    def _cfg_list(self, value, default):
        if value is None:
            return list(default)
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
            return [item for item in items if item]
        return list(value)

    def _cfg_vec3(self, value, default, device, dtype):
        if value is None:
            value = default
        if isinstance(value, str):
            values = [float(v.strip()) for v in value.split(",") if v.strip()]
        elif isinstance(value, (int, float)):
            values = [float(value)] * 3
        else:
            values = [float(v) for v in value]
        if len(values) == 1:
            values = values * 3
        if len(values) != 3:
            raise ValueError(f"Expected a scalar or 3 values, got {value}")
        return torch.tensor(values, device=device, dtype=dtype)

    def compute_lidar_intensity_for_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not all(
            key in outputs
            for key in ("rendered_reflectivity", "rendered_roughness", "rendered_normal", "rendered_intensity")
        ):
            return outputs["rendered_intensity"]
        if "lidar_viewdirs" not in image_infos:
            return outputs["rendered_intensity"]

        lidar_viewdirs = image_infos["lidar_viewdirs"].to(outputs["rendered_reflectivity"].device)
        if lidar_viewdirs.shape[:2] != outputs["rendered_reflectivity"].shape[:2]:
            return outputs["rendered_intensity"]

        lidar_ranges = image_infos.get("lidar_ranges", None)
        if lidar_ranges is None:
            lidar_ranges = torch.ones_like(outputs["rendered_reflectivity"])
        else:
            lidar_ranges = lidar_ranges.to(outputs["rendered_reflectivity"].device)

        H, W = outputs["rendered_reflectivity"].shape[:2]
        pred_lidar_intensity = rendering_equation_lidar(
            outputs["rendered_reflectivity"].reshape(-1, 1),
            outputs["rendered_roughness"].reshape(-1, 1),
            outputs["rendered_normal"].reshape(-1, 3),
            lidar_viewdirs.reshape(-1, 3),
            lidar_ranges.reshape(-1, 1),
        ).reshape(H, W, 1)

        valid_lidar_ray = torch.linalg.norm(lidar_viewdirs, dim=-1, keepdim=True) > 1e-6
        return torch.where(valid_lidar_ray, pred_lidar_intensity, outputs["rendered_intensity"])
    
    @property
    def in_test_set(self):
        return self.cur_frame.item() in self.test_set_indices
    
    def set_train(self):
        for model in self.models.values():
            model.train()
        self.train()
    
    def set_eval(self):
        for model in self.models.values():
            model.eval()
        self.eval()

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.res_schedule.downscale_times - self.step // self.res_schedule.double_steps), 0)
        else:
            return 1
        
    def update_gaussian_cfg(self, model_cfg: OmegaConf) -> OmegaConf:
        class_optim_cfg = model_cfg.get('optim', None)
        class_ctrl_cfg = model_cfg.get('ctrl', None)
        new_optim_cfg = self.gaussian_optim_general_cfg.copy()
        new_ctrl_cfg = self.gaussian_ctrl_general_cfg.copy()
        if class_optim_cfg is not None:
            new_optim_cfg.update(class_optim_cfg)
        if class_ctrl_cfg is not None:
            new_ctrl_cfg.update(class_ctrl_cfg)
        model_cfg['optim'] = new_optim_cfg
        model_cfg['ctrl'] = new_ctrl_cfg

        return model_cfg
        
    def _init_scene(self, scene_aabb) -> None:
        self.aabb = scene_aabb.to(self.device)
        scene_origin = (self.aabb[0] + self.aabb[1]) / 2
        scene_radius = torch.max(self.aabb[1] - self.aabb[0]) / 2 * 1.1
        self.scene_radius = scene_radius.item()
        self.scene_origin = scene_origin
        logger.info(f"scene origin: {scene_origin}")
        logger.info(f"scene radius: {scene_radius}")
    
    def _init_models(self) -> None:
        raise NotImplementedError("Please implement the _init_models function")
    
    def initialize_optimizer(self) -> None:
        # get param groups first
        self.param_groups = {}
        for class_name, model in self.models.items():
            self.param_groups.update(model.get_param_groups())
                 
        groups = []
        lr_schedulers = {}
        for params_name, params in self.param_groups.items():
            params = [param for param in params if param.requires_grad]
            if len(params) == 0:
                continue
            class_name = params_name.split("#")[0]
            component_name = params_name.split("#")[1]
            class_cfg = self.model_config.get(class_name)
            
            if class_cfg is not None and "optim" in class_cfg:
                class_optim_cfg = class_cfg["optim"]
                raw_optim_cfg = class_optim_cfg.get(component_name, None)
            else:
                # Fallback for models not in config (e.g. EnvMap added at runtime)
                raw_optim_cfg = {"lr": 0.001, "lr_final": 0.0001, "max_steps": self.num_iters}
            
            if raw_optim_cfg is None:
                raw_optim_cfg = {"lr": 0.001, "lr_final": 0.0001, "max_steps": self.num_iters}

            lr_scale_factor = raw_optim_cfg.get("scale_factor", 1.0)
            if isinstance(lr_scale_factor, str) and lr_scale_factor == "scene_radius":
                # scale the spatial learning rate to scene scale
                lr_scale_factor = self.scene_radius

            optim_cfg = OmegaConf.create({
                "lr": raw_optim_cfg.get('lr', 0.0005),
                "eps": raw_optim_cfg.get('eps', 1.0e-15),
                "weight_decay": raw_optim_cfg.get('weight_decay', 0),
            })
            optim_cfg.lr = optim_cfg.lr * lr_scale_factor
            assert optim_cfg is not None, f"param group {params_name} not found in config"
            lr_init = optim_cfg.lr
            groups.append({
                'params': params,
                'name': params_name,
                'lr': optim_cfg.lr,
                'eps': optim_cfg.eps,
                'weight_decay': optim_cfg.weight_decay
            })
            
            if raw_optim_cfg.get("lr_final", None) is not None:
                sched_cfg = OmegaConf.create({
                    "opt_after": raw_optim_cfg.get('opt_after', 0),
                    "warmup_steps": raw_optim_cfg.get('warmup_steps', 0),
                    "max_steps": raw_optim_cfg.get('max_steps', self.num_iters),
                    "lr_pre_warmup": raw_optim_cfg.get('lr_pre_warmup', 1.0e-8),
                    "lr_final": raw_optim_cfg.get('lr_final', None),
                    "ramp": raw_optim_cfg.get('ramp', "cosine"),
                })
                # scale the learning rate according to the scene scale
                sched_cfg.lr_pre_warmup = sched_cfg.lr_pre_warmup * lr_scale_factor
                sched_cfg.lr_final = sched_cfg.lr_final * lr_scale_factor if sched_cfg.lr_final is not None else None
                # adjust max_steps to account for opt_after
                sched_cfg.max_steps = sched_cfg.max_steps - sched_cfg.opt_after
                lr_schedulers[params_name] = lr_scheduler_fn(sched_cfg, lr_init)

        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.lr_schedulers = lr_schedulers
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.optim_general.get("use_grad_scaler", False))
        self.update_vis = False

    def update_visibility(self, update=False, sun_direction=None, mesh=False):
        if self.cur_frame.item() in self._visibility_tracings_list.keys():
            vis_num = self._visibility_tracings_list[self.cur_frame.item()].shape[0]
            pc_number = 0
            for class_name in self.gaussian_classes.keys():
                pc_number = pc_number + self.models[class_name].num_points
            if pc_number != vis_num:
                update = True

        if (self.cur_frame.item() not in self._visibility_tracings_list.keys()) or update:
            means = []
            scalings = []
            quats = []
            gaussians_inverse_covariance = []
            gaussians_opacity = []
            gaussians_normal = []
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    means.append(self.models[class_name].get_xyz)
                    scalings.append(self.models[class_name].get_scaling)
                    quats.append(self.models[class_name].get_quats)
                    gaussians_inverse_covariance.append(self.models[class_name].get_inverse_covariance())
                    gaussians_opacity.append(self.models[class_name].get_opacity[:, 0])
                    gaussians_normal.append(self.models[class_name].get_normal)
                gaussians_xyz = means = torch.concat(means)
                scalings = torch.concat(scalings)
                quats = torch.concat(quats)
                raytracer  =  RayTracer(means, scalings, quats)
                gaussians_inverse_covariance = torch.concat(gaussians_inverse_covariance)
                gaussians_opacity = torch.concat(gaussians_opacity)
                gaussians_normal = torch.concat(gaussians_normal)
                

                incident_visibility_results = []
                incident_dirs_results = []
                incident_areas_results = []
                sample_num = getattr(self, 'visibility_sample_num', 24)
                chunk_size = gaussians_xyz.shape[0]
                for offset in tqdm(range(0, gaussians_xyz.shape[0], chunk_size), "Update visibility with raytracing."):
                    incident_dirs, incident_areas = sample_incident_rays(gaussians_normal[offset:offset + chunk_size], self.training,
                                                            sample_num-1) #-1 TODO

                    if sun_direction is not None:
                        sun_direction = sun_direction
                        sun_direction = sun_direction/sun_direction.norm()
                        sun_direction = sun_direction.repeat(incident_dirs.shape[0],1,1).to(device=incident_dirs.device)
                        incident_dirs = torch.concat([sun_direction,incident_dirs], dim=1)
                        incident_areas = torch.concat([incident_areas[:,0,:].unsqueeze(1),incident_areas], dim=1)


                    trace_results = raytracer.trace_visibility(
                        gaussians_xyz[offset:offset + chunk_size, None].expand_as(incident_dirs),
                        incident_dirs,
                        gaussians_xyz,
                        gaussians_inverse_covariance,
                        gaussians_opacity,
                        gaussians_normal)
                    incident_visibility = trace_results["visibility"]
                    incident_visibility_results.append(incident_visibility)
                    incident_dirs_results.append(incident_dirs)
                    incident_areas_results.append(incident_areas)
                incident_visibility_result = torch.cat(incident_visibility_results, dim=0)
                incident_dirs_result = torch.cat(incident_dirs_results, dim=0)
                incident_areas_result = torch.cat(incident_areas_results, dim=0)
                del raytracer
                if self.cur_frame.item() in self._visibility_tracings_list:
                    del self._visibility_tracings_list[self.cur_frame.item()]
                    del self._incident_dirs_list[self.cur_frame.item()]
                    del self._incident_areas_list[self.cur_frame.item()]
                self._visibility_tracings_list.update({self.cur_frame.item(): incident_visibility_result.detach().cpu()})
                self._incident_dirs_list.update({self.cur_frame.item(): incident_dirs_result.detach().cpu()})
                self._incident_areas_list.update({self.cur_frame.item(): incident_areas_result.detach().cpu()}) 
        #del raytracer

    def update_sun_visibility(self, update=False, sun_direction=None):
        """Trace direct sun visibility only.

        The envmap image-space PBR path still needs hard direct-sun shadows, but
        it does not need the full Monte Carlo sky visibility cache used by the
        legacy per-Gaussian renderer.
        """
        if sun_direction is None:
            if "Sky" not in self.models:
                return
            sun_direction = self.models["Sky"].get_sun_direction()

        if self.cur_frame.item() in self._sun_visibility_tracings_list.keys():
            vis_num = self._sun_visibility_tracings_list[self.cur_frame.item()].shape[0]
            pc_number = 0
            for class_name in self.gaussian_classes.keys():
                pc_number = pc_number + self.models[class_name].num_points
            if pc_number != vis_num:
                update = True

        if (self.cur_frame.item() in self._sun_visibility_tracings_list.keys()) and not update:
            return

        means = []
        scalings = []
        quats = []
        gaussians_inverse_covariance = []
        gaussians_opacity = []
        gaussians_normal = []
        with torch.no_grad():
            for class_name in self.gaussian_classes.keys():
                means.append(self.models[class_name].get_xyz)
                scalings.append(self.models[class_name].get_scaling)
                quats.append(self.models[class_name].get_quats)
                gaussians_inverse_covariance.append(self.models[class_name].get_inverse_covariance())
                gaussians_opacity.append(self.models[class_name].get_opacity[:, 0])
                gaussians_normal.append(self.models[class_name].get_normal)

            gaussians_xyz = means = torch.concat(means)
            scalings = torch.concat(scalings)
            quats = torch.concat(quats)
            raytracer = RayTracer(means, scalings, quats)
            gaussians_inverse_covariance = torch.concat(gaussians_inverse_covariance)
            gaussians_opacity = torch.concat(gaussians_opacity)
            gaussians_normal = torch.concat(gaussians_normal)

            sun_direction = sun_direction.to(device=gaussians_xyz.device, dtype=gaussians_xyz.dtype)
            sun_direction = sun_direction / sun_direction.norm().clamp_min(1e-6)

            visibility_results = []
            chunk_size = int(self.render_cfg.get("sun_visibility_chunk_size", 262144))
            chunk_size = max(1, min(chunk_size, gaussians_xyz.shape[0]))
            for offset in tqdm(range(0, gaussians_xyz.shape[0], chunk_size), "Update sun visibility with raytracing."):
                dirs = sun_direction.view(1, 1, 3).expand(
                    gaussians_xyz[offset:offset + chunk_size].shape[0], 1, 3
                )
                trace_results = raytracer.trace_visibility(
                    gaussians_xyz[offset:offset + chunk_size, None].expand_as(dirs),
                    dirs,
                    gaussians_xyz,
                    gaussians_inverse_covariance,
                    gaussians_opacity,
                    gaussians_normal,
                )
                sun_visibility = trace_results["visibility"]
                if sun_visibility.dim() == 3:
                    sun_visibility = sun_visibility[:, 0, :]
                elif sun_visibility.dim() == 1:
                    sun_visibility = sun_visibility[:, None]
                visibility_results.append(sun_visibility.clamp(0.0, 1.0))

            sun_visibility_result = torch.cat(visibility_results, dim=0)
            del raytracer
            if self.cur_frame.item() in self._sun_visibility_tracings_list:
                del self._sun_visibility_tracings_list[self.cur_frame.item()]
            self._sun_visibility_tracings_list.update(
                {self.cur_frame.item(): sun_visibility_result.detach().cpu()}
            )

    def _get_current_instance_boxes(self, model):
        if not all(
            hasattr(model, attr)
            for attr in ("instances_fv", "instances_quats", "instances_trans", "instances_size")
        ):
            return None

        cur_frame = int(self.cur_frame.item())
        if cur_frame < 0 or cur_frame >= model.instances_fv.shape[0]:
            return None

        valid_mask = model.instances_fv[cur_frame].bool()
        if not valid_mask.any():
            return None

        quats_cur_frame = model.instances_quats[cur_frame]
        trans_cur_frame = model.instances_trans[cur_frame]
        if quats_cur_frame.dim() > 2:
            quats_cur_frame = quats_cur_frame[..., 0, :]

        num_frames = model.instances_fv.shape[0]
        use_interp = (
            getattr(model, "in_test_set", False)
            and cur_frame - 1 > 0
            and cur_frame + 1 < num_frames
        )
        if use_interp:
            inter_valid = (
                model.instances_fv[cur_frame - 1]
                & model.instances_fv[cur_frame + 1]
            )
            if inter_valid.any():
                quats_prev = model.instances_quats[cur_frame - 1]
                quats_next = model.instances_quats[cur_frame + 1]
                if quats_prev.dim() > 2:
                    quats_prev = quats_prev[..., 0, :]
                    quats_next = quats_next[..., 0, :]
                interp_quats = interpolate_quats(quats_prev.clone(), quats_next.clone())
                quats_cur_frame = torch.where(
                    inter_valid[:, None], interp_quats, quats_cur_frame
                )
                interp_trans = (
                    model.instances_trans[cur_frame - 1]
                    + model.instances_trans[cur_frame + 1]
                ) * 0.5
                trans_cur_frame = torch.where(
                    inter_valid[:, None], interp_trans, trans_cur_frame
                )

        quats = model.quat_act(quats_cur_frame[valid_mask])
        rotations = quat_to_rotmat(quats)
        centers = trans_cur_frame[valid_mask]
        sizes = model.instances_size[valid_mask]

        finite = (
            torch.isfinite(centers).all(dim=-1)
            & torch.isfinite(rotations).flatten(1).all(dim=-1)
            & torch.isfinite(sizes).all(dim=-1)
            & (sizes > 0).all(dim=-1)
        )
        if not finite.any():
            return None
        return centers[finite], rotations[finite], sizes[finite]

    def _collect_dynamic_sun_boxes(self, cfg, device, dtype):
        class_names = self._cfg_list(cfg.get("classes", "RigidNodes"), ["RigidNodes"])
        centers, rotations, sizes = [], [], []
        for class_name in class_names:
            model = self.models.get(class_name, None)
            if model is None:
                continue
            boxes = self._get_current_instance_boxes(model)
            if boxes is None:
                continue
            box_centers, box_rotations, box_sizes = boxes
            centers.append(box_centers.to(device=device, dtype=dtype))
            rotations.append(box_rotations.to(device=device, dtype=dtype))
            sizes.append(box_sizes.to(device=device, dtype=dtype))

        if len(centers) == 0:
            return None
        return torch.cat(centers, dim=0), torch.cat(rotations, dim=0), torch.cat(sizes, dim=0)

    @torch.no_grad()
    def compute_dynamic_box_sun_visibility(
        self,
        means_map: torch.Tensor,
        depth_map: torch.Tensor,
        opacity_map: torch.Tensor,
        sun_direction: torch.Tensor,
        dynamic_opacity_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-pixel direct-sun visibility from current-frame dynamic OBBs.

        The BVH cache traces from Gaussian centers. This image-space test traces
        from the rendered surface point at each pixel toward the sun and asks
        whether that ray intersects a dynamic object box.
        """
        cfg = self.render_cfg.get("dynamic_box_sun_visibility", {})
        if cfg is None or not cfg.get("enabled", False):
            return torch.ones_like(depth_map)

        device = means_map.device
        dtype = means_map.dtype
        boxes = self._collect_dynamic_sun_boxes(cfg, device, dtype)
        if boxes is None:
            return torch.ones_like(depth_map)

        centers, rotations, raw_sizes = boxes
        if centers.shape[0] == 0:
            return torch.ones_like(depth_map)

        size_scale = self._cfg_vec3(cfg.get("size_scale", [1.05, 1.05, 1.05]), [1.05, 1.05, 1.05], device, dtype)
        min_size = self._cfg_vec3(cfg.get("min_size", [0.0, 0.0, 0.0]), [0.0, 0.0, 0.0], device, dtype)
        hit_half_sizes = torch.maximum(raw_sizes * size_scale, min_size).clamp_min(1e-4) * 0.5

        inside_margin = float(cfg.get("inside_margin", 0.02))
        skip_half_sizes = (raw_sizes * 0.5 - inside_margin).clamp_min(0.0)
        skip_inside_boxes = bool(cfg.get("skip_inside_boxes", True))

        sun_direction = sun_direction.to(device=device, dtype=dtype)
        sun_direction = sun_direction / sun_direction.norm().clamp_min(1e-6)
        ray_dirs_local = torch.einsum("j,bjk->bk", sun_direction, rotations)

        H, W = depth_map.shape[:2]
        points = means_map.reshape(-1, 3)
        depth = depth_map.reshape(-1)
        opacity = opacity_map.reshape(-1)
        valid = (
            torch.isfinite(points).all(dim=-1)
            & torch.isfinite(depth)
            & (depth > 0)
            & (opacity > float(cfg.get("opacity_threshold", 0.01)))
        )
        if dynamic_opacity_map is not None and cfg.get("receiver_static_only", True):
            dynamic_opacity = dynamic_opacity_map.reshape(-1).to(device=device, dtype=dtype)
            valid = valid & (
                dynamic_opacity
                < float(cfg.get("receiver_dynamic_opacity_threshold", 0.2))
            )

        visibility = torch.ones(points.shape[0], 1, device=device, dtype=dtype)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return visibility.reshape(H, W, 1)

        ray_epsilon = float(cfg.get("ray_epsilon", 0.05))
        shadow_strength = float(cfg.get("shadow_strength", 1.0))
        shadow_strength = max(0.0, min(1.0, shadow_strength))
        chunk_size = int(cfg.get("chunk_size", 65536))
        chunk_size = max(1, chunk_size)

        for start in range(0, valid_indices.numel(), chunk_size):
            idx = valid_indices[start:start + chunk_size]
            pts = points[idx]
            rel = pts[:, None, :] - centers[None, :, :]
            pts_local = torch.einsum("mbj,bjk->mbk", rel, rotations)
            dirs_local = ray_dirs_local[None, :, :].expand_as(pts_local)

            parallel = dirs_local.abs() < 1e-8
            denom = torch.where(parallel, torch.ones_like(dirs_local), dirs_local)
            t0 = (-hit_half_sizes[None, :, :] - pts_local) / denom
            t1 = (hit_half_sizes[None, :, :] - pts_local) / denom
            t_near_axis = torch.minimum(t0, t1)
            t_far_axis = torch.maximum(t0, t1)

            neg_inf = torch.full_like(t_near_axis, -torch.inf)
            pos_inf = torch.full_like(t_far_axis, torch.inf)
            t_near_axis = torch.where(parallel, neg_inf, t_near_axis)
            t_far_axis = torch.where(parallel, pos_inf, t_far_axis)

            outside_parallel = parallel & (pts_local.abs() > hit_half_sizes[None, :, :])
            t_enter = t_near_axis.max(dim=-1).values
            t_exit = t_far_axis.min(dim=-1).values

            hit = (
                ~outside_parallel.any(dim=-1)
                & (t_exit > torch.maximum(
                    t_enter,
                    torch.tensor(ray_epsilon, device=device, dtype=dtype),
                ))
                & (t_exit > ray_epsilon)
            )
            if skip_inside_boxes:
                inside = (pts_local.abs() <= skip_half_sizes[None, :, :]).all(dim=-1)
                hit = hit & ~inside

            shadowed = hit.any(dim=-1)
            visibility[idx, 0] = torch.where(
                shadowed,
                torch.full((idx.shape[0],), 1.0 - shadow_strength, device=device, dtype=dtype),
                visibility[idx, 0],
            )

        return visibility.reshape(H, W, 1)

    @torch.no_grad()
    def compute_dynamic_box_contact_shadow(
        self,
        means_map: torch.Tensor,
        depth_map: torch.Tensor,
        opacity_map: torch.Tensor,
        dynamic_opacity_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Soft under-object contact shadow from current-frame dynamic OBBs.

        Direct OBB sun visibility only removes analytic sun. The region under a
        vehicle is usually dark because sky/environment light is occluded too,
        so this term is multiplied into the image-space AO map.
        """
        cfg = self.render_cfg.get("dynamic_box_sun_visibility", {})
        if cfg is None or not cfg.get("enabled", False):
            return torch.ones_like(depth_map)

        strength = float(cfg.get("contact_shadow_strength", 0.0))
        if strength <= 0:
            return torch.ones_like(depth_map)
        strength = max(0.0, min(1.0, strength))

        device = means_map.device
        dtype = means_map.dtype
        boxes = self._collect_dynamic_sun_boxes(cfg, device, dtype)
        if boxes is None:
            return torch.ones_like(depth_map)

        centers, rotations, raw_sizes = boxes
        if centers.shape[0] == 0:
            return torch.ones_like(depth_map)

        size_scale = self._cfg_vec3(
            cfg.get("contact_shadow_size_scale", cfg.get("size_scale", [1.05, 1.05, 1.05])),
            [1.05, 1.05, 1.05],
            device,
            dtype,
        )
        min_size = self._cfg_vec3(cfg.get("min_size", [0.0, 0.0, 0.0]), [0.0, 0.0, 0.0], device, dtype)
        half_sizes = torch.maximum(raw_sizes * size_scale, min_size).clamp_min(1e-4) * 0.5

        H, W = depth_map.shape[:2]
        points = means_map.reshape(-1, 3)
        depth = depth_map.reshape(-1)
        opacity = opacity_map.reshape(-1)
        valid = (
            torch.isfinite(points).all(dim=-1)
            & torch.isfinite(depth)
            & (depth > 0)
            & (opacity > float(cfg.get("opacity_threshold", 0.01)))
        )
        if dynamic_opacity_map is not None:
            dynamic_opacity = dynamic_opacity_map.reshape(-1).to(device=device, dtype=dtype)
            threshold = float(
                cfg.get(
                    "contact_shadow_dynamic_opacity_threshold",
                    cfg.get("receiver_dynamic_opacity_threshold", 0.2),
                )
            )
            valid = valid & (dynamic_opacity < threshold)

        contact = torch.ones(points.shape[0], 1, device=device, dtype=dtype)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return contact.reshape(H, W, 1)

        height = max(float(cfg.get("contact_shadow_height", 0.75)), 1e-4)
        softness = max(float(cfg.get("contact_shadow_softness", 0.35)), 1e-4)
        chunk_size = max(1, int(cfg.get("chunk_size", 65536)))

        for start in range(0, valid_indices.numel(), chunk_size):
            idx = valid_indices[start:start + chunk_size]
            rel = points[idx, None, :] - centers[None, :, :]
            pts_local = torch.einsum("mbj,bjk->mbk", rel, rotations)

            xy_excess = (pts_local[..., :2].abs() - half_sizes[None, :, :2]).clamp_min(0.0)
            footprint_fade = torch.exp(-((xy_excess / softness) ** 2).sum(dim=-1))
            bottom_distance = (pts_local[..., 2] + half_sizes[None, :, 2]).abs()
            vertical_fade = torch.exp(-((bottom_distance / height) ** 2))
            occlusion = (footprint_fade * vertical_fade).amax(dim=-1).clamp(0.0, 1.0)
            contact[idx, 0] = (1.0 - strength * occlusion).clamp(0.0, 1.0)

        return contact.reshape(H, W, 1)

    def rebuild_all_visibility(self, num_frames: int, sun_direction=None):
        """Rebuild BVH visibility caches for all frames at current model state.
        
        This ensures temporal consistency in rendered_pbr during eval.
        """
        logger.info(f"Rebuilding visibility caches for all {num_frames} frames...")
        use_image_space_pbr = self.pbr and "EnvMap" in self.models and "Sky" in self.models
        # Clear old caches
        if not use_image_space_pbr:
            self._visibility_tracings_list.clear()
            self._incident_dirs_list.clear()
            self._incident_areas_list.clear()
        self._sun_visibility_tracings_list.clear()
        old_cur_frame = self.cur_frame
        for t in range(num_frames):
            self.cur_frame = torch.tensor(t, device=self.device)
            if not use_image_space_pbr:
                self.update_visibility(update=True, sun_direction=sun_direction)
            self.update_sun_visibility(update=True, sun_direction=sun_direction)
        self.cur_frame = old_cur_frame
        logger.info("Visibility cache rebuild complete.")

    def invalidate_visibility_frames(self, frame_indices, full=True):
        """Drop stale BVH and shadow map caches for the requested frames.

        Rendering code will rebuild the missing cache lazily for exactly the
        frames it renders. This avoids rebuilding every timestep for a one-frame
        training visualization.
        """
        for frame_idx in set(int(t) for t in frame_indices):
            self._sun_visibility_tracings_list.pop(frame_idx, None)
            if full:
                self._visibility_tracings_list.pop(frame_idx, None)
                self._incident_dirs_list.pop(frame_idx, None)
                self._incident_areas_list.pop(frame_idx, None)

    def reinitialize_optimizer(self,train_sky=False,train_incident=False,train_vis=False) -> None:
        # get param groups first
        self.param_groups = {}
        class_names = self.gaussian_classes.keys()
        if train_sky:
            class_name = 'Sky'
            model = self.models[class_name]
            self.param_groups.update(model.get_param_groups())
        if train_vis:
            for class_name in class_names:
                self.param_groups.update(
                    {self.models[class_name].class_prefix+"sun_visibility": [self.models[class_name]._sun_visibility],})
        if train_incident:
            for class_name in class_names:
                self.param_groups.update(
                    {self.models[class_name].class_prefix+"incidents_dc": [self.models[class_name]._incidents_dc],
                    self.models[class_name].class_prefix+"incidents_rest": [self.models[class_name]._incidents_rest],#})
                    self.models[class_name].class_prefix+"base_color": [self.models[class_name]._base_color],})

            # self.param_groups.update({
            #     self.models['Sky'].class_prefix+"sky_intensity_scale": [self.models['Sky'].sky_intensity_scale]
            # })

        groups = []
        lr_schedulers = {}
        for params_name, params in self.param_groups.items():
            params = [param for param in params if param.requires_grad]
            if len(params) == 0:
                continue
            class_name = params_name.split("#")[0]
            component_name = params_name.split("#")[1]
            class_cfg = self.model_config.get(class_name)
            
            if class_cfg is not None and "optim" in class_cfg:
                class_optim_cfg = class_cfg["optim"]
                raw_optim_cfg = class_optim_cfg.get(component_name, None)
            else:
                raw_optim_cfg = {"lr": 0.001}
            
            if raw_optim_cfg is None:
                raw_optim_cfg = {"lr": 0.001}

            lr_scale_factor = raw_optim_cfg.get("scale_factor", 1.0)
            if isinstance(lr_scale_factor, str) and lr_scale_factor == "scene_radius":
                # scale the spatial learning rate to scene scale
                lr_scale_factor = self.scene_radius

            optim_cfg = OmegaConf.create({
                "lr": raw_optim_cfg.get('lr', 0.0005),
                "eps": raw_optim_cfg.get('eps', 1.0e-15),
                "weight_decay": raw_optim_cfg.get('weight_decay', 0),
            })
            optim_cfg.lr = optim_cfg.lr * lr_scale_factor
            assert optim_cfg is not None, f"param group {params_name} not found in config"
            lr_init = optim_cfg.lr
            groups.append({
                'params': params,
                'name': params_name,
                'lr': optim_cfg.lr,
                'eps': optim_cfg.eps,
                'weight_decay': optim_cfg.weight_decay
            })
            
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        # self.lr_schedulers = lr_schedulers
        # self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.optim_general.get("use_grad_scaler", False))
    
    def freeze_stage2_fixed_gaussian_params(self) -> None:
        """Freeze topology and fixed intrinsics while refining RGB/LiDAR albedo."""
        frozen_attrs = (
            "_means",
            "_scales",
            "_quats",
            "_opacities",
            "_features_dc",
            "_features_rest",
            "_normals",
            "_roughness",
            "_metallic",
            "_sun_visibility",
            "instances_quats",
            "instances_trans",
        )
        frozen_report = []
        trainable_report = []
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            for attr_name in frozen_attrs:
                param = getattr(model, attr_name, None)
                if param is not None:
                    param.requires_grad_(False)
                    frozen_report.append(f"{class_name}.{attr_name}")
            for attr_name in ("_base_color", "_reflectivity"):
                param = getattr(model, attr_name, None)
                if param is not None and param.requires_grad:
                    trainable_report.append(f"{class_name}.{attr_name}")
        logger.info(
            "Stage 2 frozen Gaussian params: %s",
            ", ".join(frozen_report) if frozen_report else "none",
        )
        logger.info(
            "Stage 2 trainable Gaussian material params: %s",
            ", ".join(trainable_report) if trainable_report else "none",
        )

    def validate_stage2_optimizer(self) -> None:
        """Make Stage 2 optimizer contents explicit and fail on frozen groups."""
        active_groups = [group.get("name", "unnamed") for group in self.optimizer.param_groups]
        logger.info(
            "Stage 2 active optimizer groups: %s",
            ", ".join(active_groups) if active_groups else "none",
        )
        frozen_components = {
            "xyz",
            "sh_dc",
            "sh_rest",
            "opacity",
            "scaling",
            "rotation",
            "normal",
            "metallic",
            "sun_visibility",
            "ins_rotation",
            "ins_translation",
        }
        bad_groups = []
        for group_name in active_groups:
            parts = group_name.split("#", 1)
            if len(parts) != 2:
                continue
            class_name, component_name = parts
            if class_name in self.gaussian_classes and component_name in frozen_components:
                bad_groups.append(group_name)
        if bad_groups:
            raise RuntimeError(
                "Stage 2 optimizer still contains frozen parameter groups: "
                + ", ".join(bad_groups)
            )

    def final_cull_before_freeze(self) -> None:
        """Run one last opacity/scale cull before Stage 2 freezes topology."""
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if not hasattr(model, "cull_gaussians"):
                continue
            if getattr(model, "max_2Dsize", None) is None and hasattr(model, "_means"):
                model.max_2Dsize = torch.zeros(
                    model.num_points,
                    device=model._means.device,
                    dtype=torch.float32,
                )
            n_before = model.num_points
            model.cull_gaussians()
            logger.info(
                "Stage 2 final cull %s: %d -> %d",
                class_name,
                n_before,
                model.num_points,
            )


    def _init_losses(self) -> None:
        sky_opacity_loss_fn = None
        if "Sky" in self.models:
            if self.losses_dict.mask.opacity_loss_type == "bce":
                from models.losses import binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: binary_cross_entropy(pred, gt, reduction="mean")
            elif self.losses_dict.mask.opacity_loss_type == "safe_bce":
                from models.losses import safe_binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: safe_binary_cross_entropy(pred, gt, limit=0.1, reduction="mean")
        self.sky_opacity_loss_fn = sky_opacity_loss_fn
        
        depth_loss_fn = None
        depth_loss_cfg = self.losses_dict.get("depth", None)
        if depth_loss_cfg is not None:
            from models.losses import DepthLoss
            depth_loss_fn = DepthLoss(
                loss_type=depth_loss_cfg.loss_type,
                normalize=depth_loss_cfg.normalize,
                use_inverse_depth=depth_loss_cfg.inverse_depth,
            )
        self.depth_loss_fn = depth_loss_fn
        self.region_consistency_loss = region_consistency_loss_from_labels
    
    def optimizer_zero_grad(self) -> None:
        #self.optimizer.zero_grad()
        self.optimizer.zero_grad(set_to_none=True) 
    
    def optimizer_step(self) -> None:
        self.optimizer.step()

    def preprocess_per_train_step(self, step: int) -> None:
        self.step = step
        for class_name in self.gaussian_classes.keys():
            self.models[class_name].preprocess_per_train_step(step)

        # viewer
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            self.tic = time.time()
        
    def postprocess_per_train_step(self, step: int) -> None:
        # Stage 2 follows the supplement: stop topology updates, fix geometry,
        # opacity, and fixed intrinsic params, then refine RGB/LiDAR albedo and
        # lighting with the consistency losses.
        if step >= self.freeze_step:
            if not self.freezed:
                print('Stage 2: freezing geometry, opacity, normals/roughness/metallic; training RGB/LiDAR albedo and lighting')
                self.final_cull_before_freeze()
                self.freeze_stage2_fixed_gaussian_params()
                self.initialize_optimizer()
                self.validate_stage2_optimizer()
                self.freezed = True
            if self.viewer is not None:
                self.viewer.lock.release()
            return

        # Stage 1 Gaussian post-processing: split/clone/cull/opacity reset.
        radii = self.info["radii"]
        if self.render_cfg.absgrad:
            grads = self.info["means2d"].absgrad.clone()
        else:
            grads = self.info["means2d"].grad.clone()
        if len(grads.shape)<3:
            grads = grads.unsqueeze(0)
        
        grads[..., 0] *= self.info["width"] / 2.0 * self.render_cfg.batch_size
        grads[..., 1] *= self.info["height"] / 2.0 * self.render_cfg.batch_size
        
        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
            self.models[class_name].postprocess_per_train_step(
                step=step,
                optimizer=self.optimizer,
                radii=radii[0, gaussian_mask],
                xys_grad=grads[0, gaussian_mask],
                last_size=max(self.info["width"], self.info["height"])
            )

        del grads

        # viewer
        if self.viewer is not None:
            num_train_rays_per_step = self.render_cfg.batch_size * self.info["width"] * self.info["height"]
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - self.tic)
            num_train_rays_per_sec = (
                num_train_rays_per_step * num_train_steps_per_sec
            )
            # Update the viewer state.
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            self.viewer.update(step, num_train_rays_per_step)
    
    def update_visibility_filter(self) -> None:
        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
            self.models[class_name].cur_radii = self.info["radii"][0, gaussian_mask]

    def process_camera(
        self,
        camera_infos: Dict[str, torch.Tensor],
        image_ids: torch.Tensor,
        novel_view: bool = False
    ) -> dataclass_camera:
        camtoworlds = camtoworlds_gt = camera_infos["camera_to_world"]
        
        if "CamPosePerturb" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPosePerturb"](camtoworlds, image_ids)

        if "CamPose" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPose"](camtoworlds, image_ids)
        
        height = camera_infos["height"]
        width = camera_infos["width"]
        if torch.is_tensor(height):
            height = int(height.detach().cpu().item())
        else:
            height = int(height)
        if torch.is_tensor(width):
            width = int(width.detach().cpu().item())
        else:
            width = int(width)

        # collect camera information
        camera_dict = dataclass_camera(
            camtoworlds=camtoworlds,
            camtoworlds_gt=camtoworlds_gt,
            Ks=camera_infos["intrinsics"],
            H=height,
            W=width
        )
        
        return camera_dict

    def collect_gaussians(
        self,
        cam: dataclass_camera,
        image_ids: torch.Tensor, # leave it here for future use
        sun_direction = None,
        update = False
    ) -> dataclass_gs:
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
            "class_labels": [],
        }
        
        use_image_space_pbr = self.pbr and 'EnvMap' in self.models and 'Sky' in self.models
        if sun_direction is None and 'Sky' in self.models:
            sun_direction = self.models['Sky'].get_sun_direction()

        if use_image_space_pbr:
            sun_train_start_step = int(
                self.render_cfg.get("sun_visibility_train_start_step", self.freeze_step)
            )
            use_bvh_sun_visibility = (not self.training) or (self.step >= sun_train_start_step)
            sun_update = update
            if self.training and use_bvh_sun_visibility:
                update_interval = int(self.render_cfg.get("sun_visibility_update_interval", 250))
                if update_interval > 0:
                    sun_update = sun_update or (self.step % update_interval == 0)
            if use_bvh_sun_visibility:
                self.update_sun_visibility(update=sun_update, sun_direction=sun_direction)
        else:
            if (self.step > self.freeze_step) and (self.step % 100 == 1):
                update = random.random() < 0.01   
            
            self.update_visibility(update=update, sun_direction=sun_direction)


        if self.pbr:
            gs_dict.update({"_normals":[], "_albedos":[],"_roughness":[],"_metallic":[],"_reflectivity":[],"_sun_visibility":[]})
            gs_dict.update({"_incidents":[]}) 

        for class_name in self.gaussian_classes.keys():
            gs = self.models[class_name].get_gaussians(cam)
            if gs is None:
                continue
            # collect gaussians
            gs["class_labels"] = torch.full((gs["_means"].shape[0],), self.gaussian_classes[class_name], device=self.device)
            for k, _ in gs.items():
                #if k == "_normals":
                gs_dict[k].append(gs[k])
                    
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)
            
        # get the class labels
        self.pts_labels = gs_dict.pop("class_labels")
        if self.render_dynamic_mask:
            self.dynamic_pts_mask = (self.pts_labels != 0).float()

        extras = None
        if self.pbr:
            extras = {'normals':gs_dict["_normals"],"albedos":gs_dict["_albedos"],"roughness":gs_dict["_roughness"], "metallic":gs_dict["_metallic"], "reflectivity":gs_dict["_reflectivity"], "_incidents":gs_dict["_incidents"]}
            if use_image_space_pbr:
                num_pts = gs_dict["_means"].shape[0]
                cached_sun_visibility = self._sun_visibility_tracings_list.get(self.cur_frame.item(), None)
                if cached_sun_visibility is None or cached_sun_visibility.shape[0] != num_pts:
                    sun_visibility = torch.ones(num_pts, 1, device=self.device)
                else:
                    sun_visibility = cached_sun_visibility.to(self.device)
                extras.update({
                    '_incident_dirs': torch.zeros(num_pts, 1, 3, device=self.device),
                    '_visibility_tracing': torch.ones(num_pts, 1, device=self.device),
                    '_incident_areas': torch.ones(num_pts, 1, 1, device=self.device),
                    "sun_visibility": sun_visibility})
            else:
                extras.update({
                    '_incident_dirs':self._incident_dirs_list[self.cur_frame.item()].to(self.device),
                                '_visibility_tracing':self._visibility_tracings_list[self.cur_frame.item()].to(self.device),
                                '_incident_areas':self._incident_areas_list[self.cur_frame.item()].to(self.device),
                    "sun_visibility":gs_dict["_sun_visibility"]})


        gaussians = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],    # if "means" in detach_keys, then the means will be detached
            extras=extras        # to save some extra information (TODO) more flexible way
        )
        
        return gaussians
    
    def render_gaussians(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        direct_light_env_light = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        def render_fn(opaticy_mask=None, return_info=False):
            #if gs == type? 
            color_feature = gs.rgbs
            if self.pbr:
                cam_pos = cam.camtoworlds[None, :3, 3] 
                cam_pos = cam_pos 
                viewdirs = F.normalize(cam_pos - gs.means, dim=-1)
                view_dists = torch.norm(cam_pos - gs.means, dim=1) 
                view_dists = view_dists[...,None]
                normals = gs.extras['normals']
                albedos = gs.extras['albedos']
                roughness = gs.extras['roughness']
                metallic = gs.extras['metallic']
                reflectivity = gs.extras['reflectivity']
                sun_visibility = gs.extras['sun_visibility']
                incidents = gs.extras['_incidents']

                color_feature = torch.cat([color_feature, normals, albedos, roughness, metallic, reflectivity, sun_visibility], dim=-1)

                #assert direct_light_env_light is not None
                intensity = rendering_equation_lidar(reflectivity,roughness,normals.detach(), viewdirs,view_dists.detach())

                # Skip legacy per-Gaussian MC PBR whenever the image-space
                # envmap path is active. Direct sun shadows are supplied by the
                # dedicated BVH sun-visibility cache.
                use_image_space_pbr = 'EnvMap' in self.models and 'Sky' in self.models
                if use_image_space_pbr:
                    # Dummy channels to maintain rasterizer split structure
                    brdf_color = torch.zeros_like(albedos)
                    diffuse_light = torch.zeros_like(albedos)
                    incident_sun_light = torch.zeros_like(albedos)
                else:
                    brdf_color, extra_results = rendering_equation(
                        base_color = albedos, roughness = roughness, normals = normals.detach(), viewdirs = viewdirs,
                        incidents = incidents,
                        direct_light_env_light = direct_light_env_light,
                        incident_dirs = gs.extras['_incident_dirs'],
                        incident_areas = gs.extras['_incident_areas'],
                        visibility_precompute = gs.extras['_visibility_tracing'], 
                        sun_visibility = sun_visibility, #self.sun_visibility, #,
                        xyz = gs.means,
                        step = self.step,
                        )
                    
                    # Add inference-time spotlights (e.g. headlights, street lamps)
                    if self.spotlights is not None and len(self.spotlights) > 0:
                        spotlight_color = compute_spotlight_contribution(
                            self.spotlights, gs.means, normals, albedos, roughness, viewdirs
                        )
                        brdf_color = brdf_color + spotlight_color
                    
                    diffuse_light = extra_results["diffuse_light"]
                    incident_sun_light = extra_results["incident_sun_light"]
                color_feature = torch.cat([color_feature, intensity, brdf_color, diffuse_light, incident_sun_light], dim=-1)


            renders, alphas, info = rasterization(
                means=gs.means,
                quats=gs.quats,
                scales=gs.scales,
                opacities=gs.opacities.squeeze()*opaticy_mask if opaticy_mask is not None else gs.opacities.squeeze(),
                colors=color_feature,
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
                Ks=cam.Ks[None, ...],  # [C, 3, 3]
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **kwargs,
            )
            renders = renders[0]
            alphas = alphas[0].squeeze(-1)
            assert self.render_cfg.batch_size == 1, "batch size must be 1, will support batch size > 1 in the future"

            def rasterize_opacity(point_mask):
                _, mask_alphas, _ = rasterization(
                    means=gs.means,
                    quats=gs.quats,
                    scales=gs.scales,
                    opacities=gs.opacities.squeeze() * point_mask.to(gs.opacities.device),
                    colors=torch.zeros_like(gs.rgbs),
                    viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],
                    Ks=cam.Ks[None, ...],
                    width=cam.W,
                    height=cam.H,
                    packed=self.render_cfg.packed,
                    absgrad=False,
                    sparse_grad=False,
                    rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                    **kwargs,
                )
                return mask_alphas[0].squeeze(-1)

            def rasterize_masked_render(point_mask):
                mask_renders, mask_alphas, _ = rasterization(
                    means=gs.means,
                    quats=gs.quats,
                    scales=gs.scales,
                    opacities=gs.opacities.squeeze() * point_mask.to(gs.opacities.device),
                    colors=color_feature,
                    viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],
                    Ks=cam.Ks[None, ...],
                    width=cam.W,
                    height=cam.H,
                    packed=self.render_cfg.packed,
                    absgrad=False,
                    sparse_grad=False,
                    rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                    **kwargs,
                )
                return mask_renders[0], mask_alphas[0].squeeze(-1)
            
            if self.pbr:
                rendered_rgb,rendered_normal,rendered_albedos,rendered_roughness, rendered_metallic, rendered_reflectivity, \
                rendered_sun_visibility, rendered_intensity, rendered_pbr, diffuse_light, incident_sun_light, rendered_depth = \
                 torch.split(renders, [3,3,3,1,1,1,1,1,3,3,3,1], dim=-1)              
                
                # ---- Image-space PBR with environment map ----
                # Intrinsic material properties are optimized during both
                # stages; RGB-LiDAR consistency is introduced after freeze.
                if (
                    'EnvMap' in self.models
                    and 'Sky' in self.models
                ):
                    device = renders.device
                    H, W = cam.H, cam.W
                    
                    # Compute per-pixel view directions from camera
                    # NOTE: must match rasterizer output shape [H, W]. Use indexing='ij'.
                    py, px = torch.meshgrid(
                        torch.arange(H, device=device, dtype=torch.float32),
                        torch.arange(W, device=device, dtype=torch.float32),
                        indexing='ij'
                    )
                    fx, fy = cam.Ks[0, 0], cam.Ks[1, 1]
                    cx, cy = cam.Ks[0, 2], cam.Ks[1, 2]
                    dirs = torch.stack([
                        (px - cx + 0.5) / fx,
                        (py - cy + 0.5) / fy,
                        torch.ones_like(px)
                    ], dim=-1)  # [H, W, 3]
                    # Transform to world space: dirs @ R^T
                    rays_world = dirs @ cam.camtoworlds[:3, :3].T  # [H, W, 3]
                    viewdirs_img = F.normalize(rays_world, dim=-1)
                    means_map = cam.camtoworlds[:3, 3].view(1, 1, 3) + rays_world * rendered_depth
                    
                    env_map = self.models['EnvMap']
                    sky = self.models['Sky']
                    sun_dir = sky.get_sun_direction()
                    sun_intensity = sky.sun_intensity
                    
                    # Rebuild env map mips. During training, base is optimized every step
                    # so we rebuild every step. During eval we can skip if already built.
                    # Also rebuild if device changed (e.g. CPU->CUDA after __init__).
                    need_build = self.training or getattr(env_map, 'num_mips', 0) == 0
                    if not need_build and len(env_map.specular) > 0:
                        need_build = env_map.specular[0].device != renders.device
                    if need_build:
                        env_map.build_mips()
                    
                    # Training roughness floor: prevent glossy cheat on matte surfaces.
                    # During training use a softer floor (0.15) than relighting (0.30-0.50)
                    # so cars/windows can still learn shininess, but roads stay matte.
                    train_min_roughness = getattr(self, 'relight_min_roughness', 0.15)
                    if self.training:
                        train_min_roughness = min(train_min_roughness, 0.15)

                    envmap_ao_cfg = self.render_cfg.get("envmap_ao", {})
                    if envmap_ao_cfg is None:
                        envmap_ao_cfg = {}
                    ao_map = None
                    specular_ao_strength = float(envmap_ao_cfg.get("specular_strength", 0.2))
                    if envmap_ao_cfg.get("enabled", True):
                        ao_map = compute_screen_space_ao(
                            depth_map=rendered_depth.detach(),
                            normal_map=rendered_normal.detach(),
                            opacity_map=alphas[..., None].detach(),
                            radius=int(envmap_ao_cfg.get("radius", 5)),
                            strength=float(envmap_ao_cfg.get("strength", 0.35)),
                            depth_bias=float(envmap_ao_cfg.get("depth_bias", 0.02)),
                        )

                    pbr_albedo = rendered_albedos
                    pbr_normal = rendered_normal
                    pbr_roughness = rendered_roughness
                    pbr_metallic = (
                        torch.zeros_like(rendered_metallic)
                        if (not self.training and getattr(self, "relight_force_dielectric", False))
                        else rendered_metallic
                    )
                    pbr_sun_visibility = rendered_sun_visibility
                    pbr_reflectivity = rendered_reflectivity
                    dynamic_box_sun_visibility = None
                    dynamic_box_contact_shadow = None
                    dynamic_opacity_map = None

                    dynamic_box_cfg = self.render_cfg.get("dynamic_box_sun_visibility", {})
                    if dynamic_box_cfg is None:
                        dynamic_box_cfg = {}
                    if dynamic_box_cfg.get("enabled", False):
                        contact_shadow_strength = float(dynamic_box_cfg.get("contact_shadow_strength", 0.0))
                        need_dynamic_opacity_map = (
                            dynamic_box_cfg.get("receiver_static_only", True)
                            or contact_shadow_strength > 0
                        )
                        if need_dynamic_opacity_map:
                            dynamic_point_mask = (
                                self.pts_labels != self.gaussian_classes["Background"]
                            ).float()
                            dynamic_opacity_map = rasterize_opacity(dynamic_point_mask).detach()
                        dynamic_box_sun_visibility = self.compute_dynamic_box_sun_visibility(
                            means_map=means_map.detach(),
                            depth_map=rendered_depth.detach(),
                            opacity_map=alphas[..., None].detach(),
                            sun_direction=sun_dir.detach(),
                            dynamic_opacity_map=dynamic_opacity_map,
                        )
                        pbr_sun_visibility = (pbr_sun_visibility * dynamic_box_sun_visibility).clamp(0.0, 1.0)
                        rendered_sun_visibility = pbr_sun_visibility
                        if contact_shadow_strength > 0:
                            contact_receiver = str(dynamic_box_cfg.get("contact_shadow_receiver", "full")).lower()
                            contact_means_map = means_map.detach()
                            contact_depth_map = rendered_depth.detach()
                            contact_opacity_map = alphas[..., None].detach()
                            contact_dynamic_opacity_map = dynamic_opacity_map

                            if contact_receiver in ("background", "background_only", "static"):
                                background_point_mask = (
                                    self.pts_labels == self.gaussian_classes["Background"]
                                ).float()
                                background_renders, background_alphas = rasterize_masked_render(background_point_mask)
                                background_depth = background_renders[..., -1:]
                                background_means_map = (
                                    cam.camtoworlds[:3, 3].view(1, 1, 3)
                                    + rays_world * background_depth
                                )
                                contact_means_map = background_means_map.detach()
                                contact_depth_map = background_depth.detach()
                                contact_opacity_map = background_alphas[..., None].detach()
                                contact_dynamic_opacity_map = None
                            elif contact_receiver not in ("full", "visible"):
                                raise ValueError(
                                    "dynamic_box_sun_visibility.contact_shadow_receiver "
                                    f"must be 'full' or 'background', got {contact_receiver!r}"
                                )

                            dynamic_box_contact_shadow = self.compute_dynamic_box_contact_shadow(
                                means_map=contact_means_map,
                                depth_map=contact_depth_map,
                                opacity_map=contact_opacity_map,
                                dynamic_opacity_map=contact_dynamic_opacity_map,
                            )
                            if contact_receiver in ("background", "background_only", "static") and dynamic_opacity_map is not None:
                                apply_threshold = float(
                                    dynamic_box_cfg.get(
                                        "contact_shadow_apply_dynamic_opacity_threshold",
                                        dynamic_box_cfg.get("contact_shadow_dynamic_opacity_threshold", 0.8),
                                    )
                                )
                                static_receiver = dynamic_opacity_map[..., None] < apply_threshold
                                dynamic_box_contact_shadow = torch.where(
                                    static_receiver,
                                    dynamic_box_contact_shadow,
                                    torch.ones_like(dynamic_box_contact_shadow),
                                )
                            pbr_sun_visibility = (pbr_sun_visibility * dynamic_box_contact_shadow).clamp(0.0, 1.0)
                            rendered_sun_visibility = pbr_sun_visibility
                            ao_map = (
                                dynamic_box_contact_shadow
                                if ao_map is None
                                else (ao_map * dynamic_box_contact_shadow).clamp(0.0, 1.0)
                            )

                    pbr_rgb = image_space_pbr(
                        albedo_map=pbr_albedo,
                        normal_map=pbr_normal,
                        roughness_map=pbr_roughness,
                        metallic_map=pbr_metallic,
                        sunvis_map=pbr_sun_visibility,
                        viewdir_map=viewdirs_img,
                        env_map=env_map,
                        sun_dir=sun_dir,
                        sun_intensity=sun_intensity,
                        spotlights=self.spotlights if not self.training else None,
                        depth_map=rendered_depth,
                        means_map=means_map,
                        min_roughness=train_min_roughness,
                        reflectivity_map=pbr_reflectivity,
                        ao_map=ao_map,
                        specular_ao_strength=specular_ao_strength,
                        env_diffuse_scale=float(self.render_cfg.get("env_diffuse_scale", 1.0)),
                        env_specular_scale=float(self.render_cfg.get("env_specular_scale", 1.0)),
                        env_diffuse_mode=self.render_cfg.get("env_diffuse_mode", "learned"),
                        env_ambient_floor=float(self.render_cfg.get("env_ambient_floor", 0.0)),
                    )
                    pbr_rgb = pbr_rgb * alphas[..., None]

                    if (not self.training) and self.render_cfg.get("eval_exposure_scale", 1.0) != 1.0:
                        pbr_rgb = pbr_rgb * float(self.render_cfg.get("eval_exposure_scale", 1.0))

                    rendered_pbr = pbr_rgb
                    if (not self.training) and self.render_cfg.get("eval_use_pbr_rgb", False):
                        rendered_rgb = pbr_rgb
                
                info.update({'rendered_normal':rendered_normal,
                            'rendered_albedos':rendered_albedos,
                            'rendered_roughness':rendered_roughness,
                            'rendered_metallic':rendered_metallic,
                            'rendered_reflectivity':rendered_reflectivity,
                            'rendered_pbr': rendered_pbr,
                            'diffuse_light':diffuse_light,
                            'rendered_intensity':rendered_intensity,
                            'rendered_sun_visibility':rendered_sun_visibility,
                            'incident_sun_light':incident_sun_light,
                            })
                if 'dynamic_box_sun_visibility' in locals() and dynamic_box_sun_visibility is not None:
                    info.update({
                        'rendered_dynamic_box_sun_visibility': dynamic_box_sun_visibility,
                    })
                if 'dynamic_box_contact_shadow' in locals() and dynamic_box_contact_shadow is not None:
                    info.update({
                        'rendered_dynamic_box_contact_shadow': dynamic_box_contact_shadow,
                    })

            else:
                assert renders.shape[-1] == 4, f"Must render rgb, depth and alpha"
                rendered_rgb, rendered_depth = torch.split(renders, [3, 1], dim=-1)
            
            if not return_info:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None]
            else:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None], info
        
        # render rgb and opacity
        rgb, depth, opacity, self.info = render_fn(return_info=True)
        results = {
            "rgb_gaussians": rgb,
            "depth": depth, 
            "opacity": opacity,
            #"normal": self.info['normal']
        }
        if self.pbr:
            results.update({'rendered_normal':self.info['rendered_normal'],
            'rendered_albedos':self.info['rendered_albedos'],
            'rendered_roughness':self.info['rendered_roughness'],
            'rendered_metallic':self.info['rendered_metallic'],
            'rendered_reflectivity':self.info['rendered_reflectivity'],
            'rendered_pbr':self.info['rendered_pbr'],
            'rendered_intensity':self.info['rendered_intensity'],
            'diffuse_light':self.info['diffuse_light'],
            'rendered_sun_visibility': self.info['rendered_sun_visibility'],
            'incident_sun_light':self.info['incident_sun_light'],
            })
            if 'rendered_dynamic_box_sun_visibility' in self.info:
                results.update({
                    'rendered_dynamic_box_sun_visibility': self.info['rendered_dynamic_box_sun_visibility'],
                })
            if 'rendered_dynamic_box_contact_shadow' in self.info:
                results.update({
                    'rendered_dynamic_box_contact_shadow': self.info['rendered_dynamic_box_contact_shadow'],
                })
        
        if self.training:
            self.info["means2d"].retain_grad()
        
        return results, render_fn


    def affine_transformation(
        self,
        rgb_blended: torch.Tensor,
        image_infos: Dict[str, torch.Tensor]
        ):
        if (not self.training) and self.render_cfg.get("eval_disable_affine", False):
            return rgb_blended
        if "Affine" in self.models:
            affine_trs = self.models['Affine'](image_infos)
            rgb_transformed = (affine_trs[..., :3, :3] @ rgb_blended[..., None] + affine_trs[..., :3, 3:])[..., 0]
            
            return rgb_transformed
        else:       
            return rgb_blended
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
            novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, _ = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.),
        )
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])

        rgb_blended = outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            rgb_blended, image_infos
        )
        
        return outputs
    
    def backward(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        # ----------------- backward ----------------
        total_loss = sum(loss for loss in loss_dict.values())
        self.grad_scaler.scale(total_loss).backward()
        self.optimizer_step()
        
        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            for group in self.optimizer.param_groups:
                if group["name"] in self.lr_schedulers:
                    new_lr = self.lr_schedulers[group["name"]](self.step)
                    group["lr"] = new_lr
                
    def get_loss_weight(self, loss_name, default=1.0):
        """从 config 中获取 loss 的权重，没有就返回 default"""
        loss_cfg = self.losses_dict.get(loss_name, None)
        if loss_cfg is None:
            return default
        if isinstance(loss_cfg, dict):
            return loss_cfg.get("w", default)
        elif hasattr(loss_cfg, "w"):
            return loss_cfg.w
        return loss_cfg

    def compute_rgb_detail_albedo_loss(
        self,
        rgb: torch.Tensor,
        gt_albedo: torch.Tensor,
        pred_albedo: torch.Tensor,
        valid_loss_mask: torch.Tensor,
        sky_mask: torch.Tensor,
        dynamic_mask: Optional[torch.Tensor],
        opacity: Optional[torch.Tensor],
        cfg,
    ) -> torch.Tensor:
        """Conservative stage-2 cue for bright material detail missed by RGB-X."""
        if cfg is None or not cfg.get("enabled", True):
            return pred_albedo.sum() * 0.0

        kernel_size = int(cfg.get("blur_kernel", 15))
        kernel_size = max(kernel_size, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1

        def blur_map(x: torch.Tensor) -> torch.Tensor:
            pad = kernel_size // 2
            x4 = x.permute(2, 0, 1).unsqueeze(0)
            x4 = F.pad(x4, (pad, pad, pad, pad), mode="replicate")
            x4 = F.avg_pool2d(x4, kernel_size=kernel_size, stride=1)
            return x4.squeeze(0).permute(1, 2, 0)

        with torch.no_grad():
            rgb_detached = rgb.detach().clamp(0.0, 1.0)
            prior_detached = gt_albedo.detach().clamp(0.0, 1.0)

            rgb_luma = (
                0.299 * rgb_detached[..., 0:1]
                + 0.587 * rgb_detached[..., 1:2]
                + 0.114 * rgb_detached[..., 2:3]
            )
            prior_luma = (
                0.299 * prior_detached[..., 0:1]
                + 0.587 * prior_detached[..., 1:2]
                + 0.114 * prior_detached[..., 2:3]
            )
            rgb_detail = rgb_luma - blur_map(rgb_luma)
            prior_detail = prior_luma - blur_map(prior_luma)

            rgb_max = rgb_detached.max(dim=-1, keepdim=True).values
            rgb_min = rgb_detached.min(dim=-1, keepdim=True).values
            saturation = rgb_max - rgb_min

            detail_mask = (
                (rgb_luma > float(cfg.get("luma_thresh", 0.55))).float()
                * (rgb_detail > float(cfg.get("contrast_thresh", 0.08))).float()
                * (prior_detail < float(cfg.get("prior_contrast_thresh", 0.03))).float()
                * (saturation < float(cfg.get("saturation_thresh", 0.25))).float()
                * (1.0 - sky_mask[..., None]).float()
                * valid_loss_mask[..., None].float()
            )

            if dynamic_mask is not None and cfg.get("exclude_dynamic", True):
                detail_mask = detail_mask * (1.0 - dynamic_mask[..., None]).float()
            if opacity is not None:
                opacity_thresh = float(cfg.get("opacity_thresh", 0.5))
                detail_mask = detail_mask * (opacity.detach() > opacity_thresh).float()

            target_albedo = torch.maximum(prior_detached, rgb_detached)

        # Hinge loss: only raise missing bright albedo detail, never darken it.
        per_pixel_loss = F.relu(target_albedo - pred_albedo) * detail_mask
        denom = (detail_mask.sum() * pred_albedo.shape[-1]).clamp_min(1.0)
        detail_loss = per_pixel_loss.sum() / denom
        min_pixels = float(cfg.get("min_mask_pixels", 32))
        return detail_loss * (detail_mask.sum() >= min_pixels).float()


    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # calculate loss
        loss_dict = {}

        if "egocar_masks" in image_infos:
            valid_loss_mask = (1.0 - image_infos["egocar_masks"]).float()
        else:
            valid_loss_mask = torch.ones_like(image_infos["sky_masks"])

        # ------------------------------
        # Common setup
        # ------------------------------
        gt_rgb = image_infos["pixels"] * valid_loss_mask[..., None]
        predicted_rgb = outputs["rgb"] * valid_loss_mask[..., None]

        # rgb loss
        Ll1 = torch.abs(gt_rgb - predicted_rgb).mean()
        simloss = 1 - self.ssim(gt_rgb.permute(2, 0, 1)[None, ...],
                                predicted_rgb.permute(2, 0, 1)[None, ...])
        loss_dict["rgb_loss"] = self.get_loss_weight("rgb") * Ll1
        loss_dict["ssim_loss"] = self.get_loss_weight("ssim") * simloss

        # mask loss
        gt_occupied_mask = (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask
        pred_occupied_mask = outputs["opacity"].squeeze() * valid_loss_mask
        if self.sky_opacity_loss_fn is not None:
            sky_loss_opacity = self.sky_opacity_loss_fn(pred_occupied_mask, gt_occupied_mask)
            loss_dict["sky_loss_opacity"] = self.get_loss_weight("mask") * sky_loss_opacity

        # depth loss (Stage 1 only)
        if self.depth_loss_fn is not None and self.step < self.freeze_step:
            gt_depth = image_infos["lidar_depth_map"] 
            lidar_hit_mask = (gt_depth > 0).float() * valid_loss_mask
            pred_depth = outputs["depth"]

            depth_loss = self.depth_loss_fn(pred_depth, gt_depth, lidar_hit_mask)
            lidar_w_decay = self.losses_dict.depth.get("lidar_w_decay", -1)
            if lidar_w_decay > 0:
                decay_weight = np.exp(-self.step / 8000 * lidar_w_decay)
            else:
                decay_weight = 1
            depth_loss = self.get_loss_weight("depth") * depth_loss * decay_weight
            loss_dict["depth_loss"] = depth_loss

        stage2_active = self.step > self.freeze_step

        if "rendered_pbr" in outputs:
            pbr_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
            rendered_pbr = outputs["rendered_pbr"] * pbr_mask
            gt_pbr_rgb = image_infos["pixels"] * pbr_mask
            Ll1_pbr = torch.abs(rendered_pbr - gt_pbr_rgb).mean()
            pbr_loss_name = "pbr" if stage2_active else "pbr_pre"
            loss_dict["pbr_loss"] = self.get_loss_weight(pbr_loss_name) * Ll1_pbr

        if "intensity_images" in image_infos and "rendered_intensity" in outputs:
            intensity_mask = (image_infos["intensity_images"] > 1e-3).float()
            intensity_mask = intensity_mask * valid_loss_mask[..., None]
            intensity_cfg = self.losses_dict.get("intensity", {})
            normalize_valid_pixels = bool(intensity_cfg.get("normalize_valid_pixels", False))
            intensity_weight = self.get_loss_weight("intensity")
            if isinstance(intensity_cfg, dict):
                stage_weight = intensity_cfg.get("stage2_w" if stage2_active else "stage1_w", None)
            else:
                stage_weight = getattr(intensity_cfg, "stage2_w" if stage2_active else "stage1_w", None)
            if stage_weight is not None:
                intensity_weight = stage_weight
            if intensity_mask.sum() > 0:
                predicted_lidar_intensity = self.compute_lidar_intensity_for_loss(outputs, image_infos)
                intensity_diff = torch.abs(predicted_lidar_intensity - image_infos["intensity_images"])
                if normalize_valid_pixels:
                    Ll1_intensity = (intensity_diff * intensity_mask).sum() / intensity_mask.sum().clamp_min(1.0)
                else:
                    Ll1_intensity = (intensity_diff * intensity_mask).mean()
            else:
                Ll1_intensity = outputs["rendered_intensity"].sum() * 0.0
            loss_dict["intensity_loss"] = intensity_weight * Ll1_intensity

        # Material priors are active in Stage 1. After freeze, roughness,
        # normals, metallic, and sun visibility are fixed; RGB/LiDAR albedo and
        # lighting continue to train through consistency and PBR losses. The
        # albedo prior and RGB-LiDAR consistency terms keep PBR gradients from
        # collapsing shadows into RGB albedo.
        if not stage2_active and "roughness_images" in image_infos:
            roughness_images_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
            rendered_roughness = outputs["rendered_roughness"] * roughness_images_mask
            gt_roughness = image_infos["roughness_images"] * roughness_images_mask
            L1_rough_loss = torch.abs(gt_roughness - rendered_roughness).mean()
            loss_dict["roughness_loss"] = self.get_loss_weight("roughness") * L1_rough_loss

            smooth_roughness_loss = normal_map_smooth_loss(rendered_roughness[None,...])
            loss_dict["smooth_roughness_loss"] = self.get_loss_weight("smooth_roughness") * smooth_roughness_loss

        if not stage2_active and "shading_images" in image_infos:
            images_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
            rendered_sun_visibility = outputs["rendered_sun_visibility"] * images_mask
            gt_sun_visibility = image_infos["shading_images"] * images_mask

            smooth_sun_visibility_loss = normal_map_smooth_loss(rendered_sun_visibility[None,...])
            loss_dict["smooth_sun_visibility_loss"] = self.get_loss_weight("smooth_sun_visibility") * smooth_sun_visibility_loss

            sun_visibility_loss = torch.abs(gt_sun_visibility - rendered_sun_visibility).mean()
            loss_dict["sun_visibility_loss"] = self.get_loss_weight("sun_visibility") * sun_visibility_loss

        if "albedo_images" in image_infos:
            albedo_images_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
            gt_albedo = image_infos["albedo_images"] * albedo_images_mask
            predicted_albedo = outputs["rendered_albedos"] * albedo_images_mask

            albedo_smooth_loss = normal_map_smooth_loss(predicted_albedo[None,...])
            loss_dict["albedo_smooth_loss"] = self.get_loss_weight("albedo_smooth") * albedo_smooth_loss

            Ll1_albedo = torch.abs(gt_albedo - predicted_albedo).mean()
            if stage2_active:
                loss_dict["albedo_loss"] = self.get_loss_weight("albedo") * Ll1_albedo
            else:
                loss_dict["albedo_loss"] = self.get_loss_weight("albedo_pre") * Ll1_albedo

            detail_cfg = self.losses_dict.get("rgb_detail_albedo", None)
            if detail_cfg is not None and stage2_active:
                rgb_detail_albedo_loss = self.compute_rgb_detail_albedo_loss(
                    rgb=image_infos["pixels"],
                    gt_albedo=image_infos["albedo_images"],
                    pred_albedo=outputs["rendered_albedos"],
                    valid_loss_mask=valid_loss_mask,
                    sky_mask=image_infos["sky_masks"],
                    dynamic_mask=image_infos.get("dynamic_masks", None),
                    opacity=outputs.get("opacity", None),
                    cfg=detail_cfg,
                )
                loss_dict["rgb_detail_albedo_loss"] = (
                    self.get_loss_weight("rgb_detail_albedo") * rgb_detail_albedo_loss
                )
            
            # Metallic consistency: bright albedo should be non-metallic (diffuse).
            # This prevents the model from cheating with metallic+dark_albedo on white cars.
            if not stage2_active and "rendered_metallic" in outputs and "rendered_roughness" in outputs:
                metallic_map = outputs["rendered_metallic"] * albedo_images_mask
                roughness_map = outputs["rendered_roughness"] * albedo_images_mask
                albedo_brightness = predicted_albedo.mean(dim=-1, keepdim=True)
                # Penalize metallic when albedo is bright AND roughness is high (diffuse-like)
                metallic_consistency = (metallic_map * albedo_brightness * roughness_map).mean()
                loss_dict["metallic_consistency"] = 0.05 * metallic_consistency

        if not stage2_active and "normal_images" in image_infos:
            normal_images_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
            normal_images = image_infos['normal_images'] * normal_images_mask
            predicted_normal = outputs["rendered_normal"] * normal_images_mask

            Ll1_normal_sparse = torch.abs(predicted_normal - normal_images).mean()
            loss_dict["gt_normal_loss"] = self.get_loss_weight("gt_normal") * Ll1_normal_sparse

            smooth_normal_loss = normal_map_smooth_loss(predicted_normal[None,...])
            loss_dict["smooth_normal_loss"] = self.get_loss_weight("smooth_normal") * smooth_normal_loss

        # opacity entropy reg
        if "opacity_entropy" in self.losses_dict:
            pred_opacity = torch.clamp(outputs["opacity"].squeeze(), 1e-6, 1 - 1e-6)
            loss_dict["opacity_entropy_loss"] = self.get_loss_weight("opacity_entropy") * (
                -pred_opacity * torch.log(pred_opacity)
            ).mean()

        # inverse depth smoothness reg
        if "inverse_depth_smoothness" in self.losses_dict:
            inverse_depth = 1 / (outputs["depth"] + 1e-5)
            loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(
                inverse_depth[None].repeat(1, 1, 1, 3).permute(0, 3, 1, 2),
                image_infos["pixels"][None].permute(0, 3, 1, 2)
            )
            loss_dict["inverse_depth_smoothness_loss"] = self.get_loss_weight("inverse_depth_smoothness") * loss_inv_depth

        # affine reg
        if "affine" in self.losses_dict and "Affine" in self.models:
            affine_trs = self.models['Affine']({"img_idx": image_infos["img_idx"].flatten()[0]})
            reg_mat = torch.eye(3, device=self.device)
            reg_shift = torch.zeros(3, device=self.device)
            loss_affine = torch.abs(affine_trs[..., :3, :3] - reg_mat).mean() \
                        + torch.abs(affine_trs[..., :3, 3:] - reg_shift).mean()
            loss_dict["affine_loss"] = self.get_loss_weight("affine") * loss_affine

        # ------------------------------
        # Stage 2 specific losses
        # ------------------------------
        use_ispbr_train = self.training and 'EnvMap' in self.models and 'Sky' in self.models
        if stage2_active:
            if not use_ispbr_train and "diffuse_light" in outputs:
                diffuse_light = outputs["diffuse_light"]
                mean_light = diffuse_light.mean(-1, keepdim=True).expand_as(diffuse_light)
                loss_light = F.l1_loss(diffuse_light, mean_light)
                loss_dict["diffuse_light_loss"] = self.get_loss_weight("diffuse_light") * loss_light

            if "intensity_images" in image_infos and "albedo_images" in image_infos:
                images_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
                rendered_reflectivity = outputs["rendered_reflectivity"] * images_mask
                gt_albedo = image_infos["albedo_images"] * images_mask

                albedo_mean = gt_albedo.mean(dim=-1)
                ref_neighborhood_smoothness_loss = neighborhood_smoothness_loss(albedo_mean[...,None], rendered_reflectivity)
                loss_dict["ref_neighborhood_smoothness_loss"] = self.get_loss_weight("ref_neighborhood_smoothness") * ref_neighborhood_smoothness_loss

                region_weight = self.get_loss_weight("ref_region_consistency_albedo", default=0.0)
                if region_weight > 0:
                    if "region_labels" not in image_infos:
                        raise RuntimeError(
                            "ref_region_consistency_albedo requires cached region labels. "
                            "Run tools/precompute_reflectivity_sam_regions.py and train with "
                            "data.pixel_source.load_region_maps=true, or set "
                            "trainer.losses.ref_region_consistency_albedo.w=0."
                        )
                    region_cfg = self.losses_dict.get("ref_region_consistency_albedo", {})
                    region_valid_mask = (1 - image_infos["sky_masks"]) * valid_loss_mask
                    if "dynamic_masks" in image_infos:
                        region_valid_mask = region_valid_mask * (1 - image_infos["dynamic_masks"])
                    ref_region_consistency_loss = self.region_consistency_loss(
                        image_infos["region_labels"],
                        outputs["rendered_albedos"],
                        region_valid_mask,
                        min_region_pixels=int(region_cfg.get("min_region_pixels", 3)),
                    )
                    loss_dict["ref_region_consistency_loss"] = region_weight * ref_region_consistency_loss

        # dynamic region loss
        dynamic_region_cfg = self.losses_dict.get("dynamic_region", None)
        if dynamic_region_cfg is not None:
            weight_factor = dynamic_region_cfg.get("w", 1.0)
            start_from = dynamic_region_cfg.get("start_from", 0)
            if self.step == start_from:
                self.render_dynamic_mask = True
            if self.step > start_from and "Dynamic_opacity" in outputs:
                dynamic_pred_mask = (outputs["Dynamic_opacity"].data > 0.2).squeeze()
                dynamic_pred_mask = dynamic_pred_mask & valid_loss_mask.bool()
                if dynamic_pred_mask.sum() > 0:
                    Ll1 = torch.abs(gt_rgb[dynamic_pred_mask] - predicted_rgb[dynamic_pred_mask]).mean()
                    loss_dict["vehicle_region_rgb_loss"] = weight_factor * Ll1

        # gaussian reg losses
        for class_name in self.gaussian_classes.keys():
            class_reg_loss = self.models[class_name].compute_reg_loss()
            for k, v in class_reg_loss.items():
                loss_dict[f"{class_name}_{k}"] = v

        return loss_dict

    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        metric_dict = {}
        psnr = self.psnr(outputs["rgb"], image_infos["pixels"])
        metric_dict.update({"psnr": psnr})
        return metric_dict
    
    def get_gaussian_count(self):
        num_dict = {}
        for class_name in self.gaussian_classes.keys():
            num_dict[class_name] = self.models[class_name].num_points
        return num_dict
    
    def state_dict(self, only_model: bool = True):
        state_dict = super().state_dict()
        state_dict.update({
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "step": self.step,
        })
        if not only_model:
            state_dict.update({
                "optimizer": {k: v.state_dict() for k, v in self.optimizer.items()},
            })
        return state_dict

    def load_state_dict(self, state_dict: dict, load_only_model: bool =True, strict: bool = True):
        step = state_dict.pop("step")
        self.step = step
        logger.info(f"Loading checkpoint at step {step}")

        # load optimizer and schedulers
        if "optimizer" in state_dict:
            loaded_state_optimizers = state_dict.pop("optimizer")
        # if "schedulers" in state_dict:
        #     loaded_state_schedulers = state_dict.pop("schedulers")
        # if "grad_scaler" in state_dict:
        #     loaded_grad_scaler = state_dict.pop("grad_scaler")
        if not load_only_model:
            raise NotImplementedError("Now only support loading model, \
                it seems there is no need to load optimizer and schedulers")
            for k, v in loaded_state_optimizers.items():
                self.optimizer[k].load_state_dict(v)
            for k, v in loaded_state_schedulers.items():
                self.schedulers[k].load_state_dict(v)
            self.grad_scaler.load_state_dict(loaded_grad_scaler)
        
        # load model
        model_state_dict = state_dict.pop("models")
        for class_name in self.models.keys():
            # if class_name == 'Sky': #TODO
            #     continue
            model = self.models[class_name]
            model.step = step
            if class_name not in model_state_dict:
                if class_name in self.gaussian_classes:
                    self.gaussian_classes.pop(class_name)
                logger.warning(f"Cannot find {class_name} in the checkpoint")
                continue
            msg = model.load_state_dict(model_state_dict[class_name], strict=strict)
            logger.info(f"{class_name}: {msg}")
        msg = super().load_state_dict(state_dict, strict)
        logger.info(f"BasicTrainer: {msg}")
        if self.step >= self.freeze_step:
            logger.info("Checkpoint step is in Stage 2; applying freeze schedule before optimizer setup")
            self.freeze_stage2_fixed_gaussian_params()
            self.initialize_optimizer()
            self.validate_stage2_optimizer()
            self.freezed = True
        else:
            self.freezed = False
        
    def resume_from_checkpoint(
        self,
        ckpt_path: str,
        load_only_model: bool=True
    ) -> None:
        """
        Load model from checkpoint.
        """
        logger.info(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path,weights_only=False)
        self.load_state_dict(state_dict, load_only_model=load_only_model, strict=False)
        
    def save_checkpoint(
        self,
        log_dir: str,
        save_only_model: bool=True,
        is_final: bool=False
    ) -> None:
        """
        Save model to checkpoint.
        """
        if is_final:
            ckpt_path = os.path.join(log_dir, f"checkpoint_final.pth")
        else:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{self.step:05d}.pth")
        torch.save(self.state_dict(only_model=save_only_model), ckpt_path)
        logger.info(f"Saved a checkpoint to {ckpt_path}")
        
    def init_viewer(self, port: int = 8080):
        # a simple viewer for background ONLY visualization
        self.server = viser.ViserServer(port=port, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            mode="training",
        )

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer. Uses the full trainer forward pass."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        # Build meshgrid for ray generation
        x, y = torch.meshgrid(
            torch.arange(W, device=self.device),
            torch.arange(H, device=self.device),
            indexing="xy",
        )
        x, y = x.flatten(), y.flatten()

        # Compute ray directions (viewdirs) and origins
        origins, viewdirs, _ = get_rays(x, y, c2w, K)
        origins = origins.reshape(H, W, 3)
        viewdirs = viewdirs.reshape(H, W, 3)

        # Normalized pixel coordinates for Affine model
        pixel_coords = torch.stack([y.float() / H, x.float() / W], dim=-1).reshape(H, W, 2)

        # Default to first frame for temporal state
        default_normed_time = self.normalized_timestamps[0] if len(self.normalized_timestamps) > 0 else 0.0
        normed_time = torch.full((H, W), default_normed_time, dtype=torch.float32, device=self.device)
        img_idx = torch.zeros((H, W), dtype=torch.long, device=self.device)
        frame_idx = torch.zeros((H, W), dtype=torch.long, device=self.device)

        image_infos = {
            "origins": origins,
            "viewdirs": viewdirs,
            "pixel_coords": pixel_coords,
            "normed_time": normed_time,
            "img_idx": img_idx,
            "frame_idx": frame_idx,
        }

        camera_infos = {
            "camera_to_world": c2w,
            "intrinsics": K,
            "height": torch.tensor(H, dtype=torch.long, device=self.device),
            "width": torch.tensor(W, dtype=torch.long, device=self.device),
        }

        outputs = self(image_infos, camera_infos, novel_view=True)
        rgb = outputs["rgb"].clamp(0.0, 1.0)
        return rgb.cpu().numpy()
