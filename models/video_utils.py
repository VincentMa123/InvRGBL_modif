from typing import Literal, Dict, List, Optional, Callable
from tqdm import tqdm, trange
import numpy as np
import os
import logging
import imageio
from IPython import embed
import torch
from torch import Tensor
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim

try:
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from datasets.base import SplitWrapper
from models.trainers.base import BasicTrainer
from utils.visualization import (
    to8b,
    depth_visualizer,
)

logger = logging.getLogger()

def get_numpy(x: Tensor) -> np.ndarray:
    return x.squeeze().cpu().numpy()


def apply_colormap(frame: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """Apply a matplotlib colormap to a grayscale or scalar frame.

    Args:
        frame: (H, W) or (H, W, 1) or (H, W, 3) numpy array in [0, 1].
        cmap_name: Matplotlib colormap name (e.g., "viridis", "plasma", "jet", "turbo").

    Returns:
        (H, W, 3) RGB uint8 image.
    """
    if not HAS_MPL:
        # Fallback: replicate grayscale to RGB
        if frame.ndim == 3 and frame.shape[-1] == 3:
            return to8b(frame)
        gray = frame.squeeze()
        return to8b(np.stack([gray, gray, gray], axis=-1))

    # Extract scalar value
    if frame.ndim == 3 and frame.shape[-1] == 3:
        # Already RGB — assume grayscale replicated; take first channel
        scalar = frame[..., 0]
    else:
        scalar = frame.squeeze()

    cmap = cm.get_cmap(cmap_name)
    colored = cmap(np.clip(scalar, 0.0, 1.0))[..., :3]  # drop alpha
    return to8b(colored)

def apply_black_mask(frame: np.ndarray, mask: Optional[Tensor]) -> np.ndarray:
    if mask is None:
        return frame
    mask_np = mask.squeeze().detach().cpu().numpy().astype(bool)
    out = frame.copy()
    out[mask_np] = 0
    return out

def non_zero_mean(x: Tensor) -> float:
    return sum(x) / len(x) if len(x) > 0 else -1

def compute_psnr(prediction: Tensor, target: Tensor) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between the prediction and target tensors.

    Args:
        prediction (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The PSNR value between the prediction and target tensors.
    """
    if not isinstance(prediction, Tensor):
        prediction = Tensor(prediction)
    if not isinstance(target, Tensor):
        target = Tensor(target).to(prediction.device)
    return (-10 * torch.log10(F.mse_loss(prediction, target))).item()


def render_images(
    trainer: BasicTrainer,
    dataset: SplitWrapper,
    compute_metrics: bool = False,
    compute_error_map: bool = False,
    vis_indices: Optional[List[int]] = None
):
    """
    Render pixel-related outputs from a model.

    Args:
        ....skip obvious args
        compute_metrics (bool, optional): Whether to compute metrics. Defaults to False.
        vis_indices (Optional[List[int]], optional): Indices to visualize. Defaults to None.
    """
    trainer.set_eval()
    if hasattr(trainer, "invalidate_visibility_frames"):
        indices = vis_indices if vis_indices is not None else range(len(dataset))
        rendered_frame_indices = []
        num_cams = getattr(dataset.datasource, "num_cams", 1)
        for local_idx in indices:
            global_idx = dataset.split_indices[local_idx]
            rendered_frame_indices.append(global_idx // num_cams)
        use_image_space_pbr = trainer.pbr and "EnvMap" in trainer.models and "Sky" in trainer.models
        trainer.invalidate_visibility_frames(
            rendered_frame_indices,
            full=not use_image_space_pbr,
        )
    render_results = render(
        dataset,
        trainer=trainer,
        compute_metrics=compute_metrics,
        compute_error_map=compute_error_map,
        vis_indices=vis_indices
    )
    if compute_metrics:
        num_samples = len(dataset) if vis_indices is None else len(vis_indices)
        logger.info(f"Eval over {num_samples} images:")
        logger.info(f"\t Full Image  PSNR: {render_results['psnr']:.4f}")
        logger.info(f"\t Full Image  SSIM: {render_results['ssim']:.4f}")
        logger.info(f"\t Full Image LPIPS: {render_results['lpips']:.4f}")
        logger.info(f"\t LiDAR RMSE: {render_results['rmse']:.4f}")
        logger.info(f"\t     Non-Sky PSNR: {render_results['occupied_psnr']:.4f}")
        logger.info(f"\t     Non-Sky SSIM: {render_results['occupied_ssim']:.4f}")
        logger.info(f"\tDynamic-Only PSNR: {render_results['masked_psnr']:.4f}")
        logger.info(f"\tDynamic-Only SSIM: {render_results['masked_ssim']:.4f}")
        logger.info(f"\t  Human-Only PSNR: {render_results['human_psnr']:.4f}")
        logger.info(f"\t  Human-Only SSIM: {render_results['human_ssim']:.4f}")
        logger.info(f"\tVehicle-Only PSNR: {render_results['vehicle_psnr']:.4f}")
        logger.info(f"\tVehicle-Only SSIM: {render_results['vehicle_ssim']:.4f}")

    return render_results


def render(
    dataset: SplitWrapper,
    trainer: BasicTrainer = None,
    compute_metrics: bool = False,
    compute_error_map: bool = False,
    vis_indices: Optional[List[int]] = None,
):
    """
    Renders a dataset utilizing a specified render function.

    Parameters:
        dataset: Dataset to render.
        trainer: Gaussian trainer, includes gaussian models and rendering modules
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
        compute_error_map: Optional; if True, the function will compute and return error maps. Default is False.
        vis_indices: Optional; if not None, the function will only render the specified indices. Default is None.
    """
    # rgbs
    rgbs, gt_rgbs, rgb_sky_blend, rgb_sky = [], [], [], []
    Background_rgbs, RigidNodes_rgbs, DeformableNodes_rgbs, SMPLNodes_rgbs, Dynamic_rgbs = [], [], [], [], []
    error_maps = []

    # depths
    depths, lidar_on_images = [], []
    Background_depths, RigidNodes_depths, DeformableNodes_depths, SMPLNodes_depths, Dynamic_depths = [], [], [], [], []
    
    gt_normals = []
    gt_albedos = []
    # normal
    normals = []
    albedos = []
    rendered_normals = []
    pbrs = []
    pbrs_layered = []
    pbrs_background = []
    pbrs_dynamic = []
    rendered_pbr_dynamic_composite_mask = []
    rendered_roughness =[]
    rendered_metallic = []
    rendered_reflectivity= []
    rendered_intensity = []
    rendered_sun_visibility = []
    rendered_dynamic_box_sun_visibility = []
    rendered_dynamic_box_contact_shadow = []
    rendered_dynamic_box_contact_shadow_raw = []
    rendered_dynamic_box_contact_apply_mask = []
    rendered_dynamic_opacity = []
    # sky
    opacities, sky_masks = [], []
    Background_opacities, RigidNodes_opacities, DeformableNodes_opacities, SMPLNodes_opacities, Dynamic_opacities = [], [], [], [], []
    diffuse_lights = []
    # misc
    cam_names, cam_ids = [], []

    gt_intensitys = [] 

    if compute_metrics:
        psnrs, ssim_scores, lpipss, RMSEs = [], [], [], []
        masked_psnrs, masked_ssims = [], []
        human_psnrs, human_ssims = [], []
        vehicle_psnrs, vehicle_ssims = [], []
        occupied_psnrs, occupied_ssims = [], []

    with torch.no_grad():
        indices = vis_indices if vis_indices is not None else range(len(dataset))
        camera_downscale = trainer._get_downscale_factor()
        for i in tqdm(indices, desc=f"rendering {dataset.split}", dynamic_ncols=True):
            # get image and camera infos
            image_infos, cam_infos = dataset.get_image(i, camera_downscale)
            for k, v in image_infos.items():
                if isinstance(v, Tensor):
                    image_infos[k] = v.cuda(non_blocking=True)
            for k, v in cam_infos.items():
                if isinstance(v, Tensor):
                    cam_infos[k] = v.cuda(non_blocking=True)
            # render the image
            results = trainer(image_infos, cam_infos)
            # --- debug: log maps at freeze step ---
            try:
                step_val = getattr(trainer, 'step', None)
            except Exception:
                step_val = None
            if step_val == 15000:
                def stats(x):
                    if not isinstance(x, Tensor):
                        return None
                    t = x.detach()
                    return dict(min=float(t.min()), max=float(t.max()), mean=float(t.mean()), zeros=int((t==0).sum()), nans=int(torch.isnan(t).sum()))

                keys_to_check = ["rendered_intensity", "rendered_sun_visibility", "rendered_normal", "rendered_roughness", "rendered_reflectivity"]
                debug_stats = {}
                for k in keys_to_check:
                    if k in results:
                        debug_stats[k] = stats(results[k])
                logger.info(f"[DEBUG][step={step_val}] rendered maps stats: {debug_stats}")
            
            # ------------- clip rgb ------------- #
            for k, v in results.items():
                if isinstance(v, Tensor) and "rgb" in k:
                    results[k] = v.clamp(0., 1.)
            
            # ------------- cam names ------------- #
            cam_names.append(cam_infos["cam_name"])
            cam_ids.append(
                cam_infos["cam_id"].flatten()[0].cpu().numpy()
            )

            # ------------- rgb ------------- #
            rgb = results["rgb"]
            rgbs.append(get_numpy(rgb))
            if "pixels" in image_infos:
                gt_rgbs.append(get_numpy(image_infos["pixels"]))

            if "normal_images" in image_infos:
                gt_normals.append((get_numpy(image_infos["normal_images"])+1)/2)

            if "albedo_images" in image_infos:
                gt_albedos.append(get_numpy(image_infos["albedo_images"]))

            green_background = torch.tensor([0.0, 177, 64]) / 255.0
            green_background = green_background.to(rgb.device)
            if "Background_rgb" in results:
                Background_rgb = results["Background_rgb"] * results[
                    "Background_opacity"
                ] + green_background * (1 - results["Background_opacity"])
                if "sky_masks" in image_infos:
                    sky_mask_vis = image_infos["sky_masks"].unsqueeze(-1)
                    Background_rgb = Background_rgb * (1 - sky_mask_vis) + green_background * sky_mask_vis
                Background_rgbs.append(get_numpy(Background_rgb))
            if "RigidNodes_rgb" in results:
                RigidNodes_rgb = results["RigidNodes_rgb"] * results[
                    "RigidNodes_opacity"
                ] + green_background * (1 - results["RigidNodes_opacity"])
                RigidNodes_rgbs.append(get_numpy(RigidNodes_rgb))
            if "DeformableNodes_rgb" in results:
                DeformableNodes_rgb = results["DeformableNodes_rgb"] * results[
                    "DeformableNodes_opacity"
                ] + green_background * (1 - results["DeformableNodes_opacity"])
                DeformableNodes_rgbs.append(get_numpy(DeformableNodes_rgb))
            if "SMPLNodes_rgb" in results:
                SMPLNodes_rgb = results["SMPLNodes_rgb"] * results[
                    "SMPLNodes_opacity"
                ] + green_background * (1 - results["SMPLNodes_opacity"])
                SMPLNodes_rgbs.append(get_numpy(SMPLNodes_rgb))
            if "Dynamic_rgb" in results:
                Dynamic_rgb = results["Dynamic_rgb"] * results[
                    "Dynamic_opacity"
                ] + green_background * (1 - results["Dynamic_opacity"])
                Dynamic_rgbs.append(get_numpy(Dynamic_rgb))
            if compute_error_map:
                # cal mean squared error
                error_map = (rgb - image_infos["pixels"]) ** 2
                error_map = error_map.mean(dim=-1, keepdim=True)
                # scale
                error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
                error_map = error_map.repeat_interleave(3, dim=-1)
                error_maps.append(get_numpy(error_map))
            if "rgb_sky_blend" in results:
                rgb_sky_blend.append(get_numpy(results["rgb_sky_blend"]))
            if "rgb_sky" in results:
                rgb_sky.append(get_numpy(results["rgb_sky"]))
            # ------------- depth ------------- #
            depth = results["depth"]
            depths.append(get_numpy(depth))
            #--------------normal----------------#
            # For material visualizations, mask sky regions so unconstrained sky Gaussians
            # don't show as black artifacts.
            sky_mask_vis = None
            if "sky_masks" in image_infos:
                sky_mask_vis = image_infos["sky_masks"].unsqueeze(-1)  # [H, W, 1]

            if "normal" in results:
                normal = results["normal"]
                normal_vis = (get_numpy(normal) + 1) / 2
                normals.append(apply_black_mask(normal_vis, sky_mask_vis))
            if "rendered_albedos" in results:
                albedo = results["rendered_albedos"]
                if sky_mask_vis is not None:
                    albedo = albedo * (1 - sky_mask_vis)
                albedos.append(get_numpy(albedo))
            if "rendered_roughness" in results:
                roughness = results["rendered_roughness"]
                roughness_vis = apply_colormap(get_numpy(roughness), cmap_name="viridis")
                rendered_roughness.append(apply_black_mask(roughness_vis, sky_mask_vis))
            if "rendered_metallic" in results:
                metallic = results["rendered_metallic"]
                metallic_vis = apply_colormap(get_numpy(metallic), cmap_name="plasma")
                rendered_metallic.append(apply_black_mask(metallic_vis, sky_mask_vis))
            if "rendered_reflectivity" in results:
                reflectivity = results["rendered_reflectivity"]
                reflectivity_vis = apply_colormap(get_numpy(reflectivity), cmap_name="viridis")
                rendered_reflectivity.append(apply_black_mask(reflectivity_vis, sky_mask_vis))

            if "rendered_intensity" in results:
                intensity = results["rendered_intensity"]
                intensity_vis = apply_colormap(get_numpy(intensity), cmap_name="turbo")
                rendered_intensity.append(apply_black_mask(intensity_vis, sky_mask_vis))

            if "rendered_sun_visibility" in results:
                sun_visibility = results["rendered_sun_visibility"]
                rendered_sun_visibility.append(apply_colormap(get_numpy(sun_visibility), cmap_name="gray")) 
            if "rendered_dynamic_box_sun_visibility" in results:
                box_sun_visibility = results["rendered_dynamic_box_sun_visibility"]
                rendered_dynamic_box_sun_visibility.append(
                    apply_colormap(get_numpy(box_sun_visibility), cmap_name="gray")
                )
            if "rendered_dynamic_box_contact_shadow" in results:
                contact_shadow = results["rendered_dynamic_box_contact_shadow"]
                rendered_dynamic_box_contact_shadow.append(
                    apply_colormap(get_numpy(contact_shadow), cmap_name="gray")
                )
            if "rendered_dynamic_box_contact_shadow_raw" in results:
                contact_shadow_raw = results["rendered_dynamic_box_contact_shadow_raw"]
                rendered_dynamic_box_contact_shadow_raw.append(
                    apply_colormap(get_numpy(contact_shadow_raw), cmap_name="gray")
                )
            if "rendered_dynamic_box_contact_apply_mask" in results:
                contact_apply_mask = results["rendered_dynamic_box_contact_apply_mask"]
                rendered_dynamic_box_contact_apply_mask.append(
                    apply_colormap(get_numpy(contact_apply_mask), cmap_name="gray")
                )
            if "rendered_dynamic_opacity" in results:
                dynamic_opacity = results["rendered_dynamic_opacity"]
                rendered_dynamic_opacity.append(
                    apply_colormap(get_numpy(dynamic_opacity), cmap_name="gray")
                )

            if "rendered_pbr" in results:
                pbr = results["rendered_pbr"]
                pbrs.append(get_numpy(pbr))      
            if "rendered_pbr_layered" in results:
                pbr_layered = results["rendered_pbr_layered"]
                pbrs_layered.append(get_numpy(pbr_layered))
            if "rendered_pbr_background" in results:
                pbr_background = results["rendered_pbr_background"]
                pbrs_background.append(get_numpy(pbr_background))
            if "rendered_pbr_dynamic" in results:
                pbr_dynamic = results["rendered_pbr_dynamic"]
                pbrs_dynamic.append(get_numpy(pbr_dynamic))
            if "rendered_pbr_dynamic_composite_mask" in results:
                dynamic_composite_mask = results["rendered_pbr_dynamic_composite_mask"]
                rendered_pbr_dynamic_composite_mask.append(
                    apply_colormap(get_numpy(dynamic_composite_mask), cmap_name="gray")
                )
            
            if "intensity_images" in image_infos:
                gt_intensity = image_infos["intensity_images"]
                gt_intensity_np = get_numpy(gt_intensity)
                gt_intensity_vis = apply_colormap(gt_intensity_np, cmap_name="turbo")
                gt_intensity_vis[np.squeeze(gt_intensity_np) <= 0] = 0
                gt_intensitys.append(apply_black_mask(gt_intensity_vis, sky_mask_vis))

            if "diffuse_light" in results:
                diffuse_light = results["diffuse_light"]
                max_light = diffuse_light.max()
                if max_light > 0:
                    diffuse_light = diffuse_light / max_light
                diffuse_lights.append(get_numpy(diffuse_light))    

            if "rendered_normal" in results:
                rendered_normal = results["rendered_normal"]
                n = torch.zeros_like(rendered_normal)
                n[...,0] = - rendered_normal[...,1]
                n[...,1] = - rendered_normal[...,2]
                n[...,2] =  rendered_normal[...,0]
                rendered_normal_vis = (get_numpy(n) + 1) / 2
                rendered_normals.append(apply_black_mask(rendered_normal_vis, sky_mask_vis))

            # ------------- mask ------------- #
            if "opacity" in results:
                opacities.append(get_numpy(results["opacity"]))
            if "Background_depth" in results:
                Background_depths.append(get_numpy(results["Background_depth"]))
                Background_opacities.append(get_numpy(results["Background_opacity"]))
            if "RigidNodes_depth" in results:
                RigidNodes_depths.append(get_numpy(results["RigidNodes_depth"]))
                RigidNodes_opacities.append(get_numpy(results["RigidNodes_opacity"]))
            if "DeformableNodes_depth" in results:
                DeformableNodes_depths.append(get_numpy(results["DeformableNodes_depth"]))
                DeformableNodes_opacities.append(get_numpy(results["DeformableNodes_opacity"]))
            if "SMPLNodes_depth" in results:
                SMPLNodes_depths.append(get_numpy(results["SMPLNodes_depth"]))
                SMPLNodes_opacities.append(get_numpy(results["SMPLNodes_opacity"]))
            if "Dynamic_depth" in results:
                Dynamic_depths.append(get_numpy(results["Dynamic_depth"]))
                Dynamic_opacities.append(get_numpy(results["Dynamic_opacity"]))
            if "sky_masks" in image_infos:
                sky_masks.append(get_numpy(image_infos["sky_masks"]))
                
            # ------------- lidar ------------- #
            if "lidar_depth_map" in image_infos:
                depth_map = image_infos["lidar_depth_map"]
                depth_img = depth_map.cpu().numpy()
                depth_img = depth_visualizer(depth_img, depth_img > 0)
                mask = (depth_map.unsqueeze(-1) > 0).cpu().numpy()
                lidar_on_image = image_infos["pixels"].cpu().numpy() * (1 - mask) + depth_img * mask
                lidar_on_images.append(lidar_on_image)

            if compute_metrics:
                psnr = compute_psnr(rgb, image_infos["pixels"])
                ssim_score = ssim(
                    get_numpy(rgb),
                    get_numpy(image_infos["pixels"]),
                    data_range=1.0,
                    channel_axis=-1,
                )
                lpips = trainer.lpips(
                    rgb[None, ...].permute(0, 3, 1, 2),
                    image_infos["pixels"][None, ...].permute(0, 3, 1, 2)
                )
                logger.info(f"Frame {i}: PSNR {psnr:.4f}, SSIM {ssim_score:.4f}")
                psnrs.append(psnr)
                ssim_scores.append(ssim_score)
                lpipss.append(lpips.item())

                lidar_mask = (image_infos['intensity_images'] > 5e-2)
                rendered_intensity_ = results["rendered_intensity"] * lidar_mask
                intensity_images = image_infos['intensity_images'] * lidar_mask
                RMSE = torch.sqrt(torch.sum((rendered_intensity_ - intensity_images) ** 2)/lidar_mask.sum())
                RMSEs.append(RMSE.item())
                # array = intensity_images.cpu().numpy() * 255
                # array = array.astype(np.uint8)
                # array_2d = array.squeeze()
                # img = Image.fromarray(array_2d, mode='L')  # 'L' mode for grayscale images
                # img.save("output_image1.png")



                if "sky_masks" in image_infos:
                    occupied_mask = ~get_numpy(image_infos["sky_masks"]).astype(bool)
                    if occupied_mask.sum() > 0:
                        occupied_psnrs.append(
                            compute_psnr(
                                rgb[occupied_mask], image_infos["pixels"][occupied_mask]
                            )
                        )
                        occupied_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][occupied_mask].mean()
                        )

                if "dynamic_masks" in image_infos:
                    dynamic_mask = get_numpy(image_infos["dynamic_masks"]).astype(bool)
                    if dynamic_mask.sum() > 0:
                        masked_psnrs.append(
                            compute_psnr(
                                rgb[dynamic_mask], image_infos["pixels"][dynamic_mask]
                            )
                        )
                        masked_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][dynamic_mask].mean()
                        )
                
                if "human_masks" in image_infos:
                    human_mask = get_numpy(image_infos["human_masks"]).astype(bool)
                    if human_mask.sum() > 0:
                        human_psnrs.append(
                            compute_psnr(
                                rgb[human_mask], image_infos["pixels"][human_mask]
                            )
                        )
                        human_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][human_mask].mean()
                        )
                
                if "vehicle_masks" in image_infos:
                    vehicle_mask = get_numpy(image_infos["vehicle_masks"]).astype(bool)
                    if vehicle_mask.sum() > 0:
                        vehicle_psnrs.append(
                            compute_psnr(
                                rgb[vehicle_mask], image_infos["pixels"][vehicle_mask]
                            )
                        )
                        vehicle_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][vehicle_mask].mean()
                        )

    # messy aggregation...
    results_dict = {}
    results_dict["psnr"] = non_zero_mean(psnrs) if compute_metrics else -1
    results_dict["ssim"] = non_zero_mean(ssim_scores) if compute_metrics else -1
    results_dict["lpips"] = non_zero_mean(lpipss) if compute_metrics else -1
    results_dict["rmse"] = non_zero_mean(RMSEs) if compute_metrics else -1
    results_dict["occupied_psnr"] = non_zero_mean(occupied_psnrs) if compute_metrics else -1
    results_dict["occupied_ssim"] = non_zero_mean(occupied_ssims) if compute_metrics else -1
    results_dict["masked_psnr"] = non_zero_mean(masked_psnrs) if compute_metrics else -1
    results_dict["masked_ssim"] = non_zero_mean(masked_ssims) if compute_metrics else -1
    results_dict["human_psnr"] = non_zero_mean(human_psnrs) if compute_metrics else -1
    results_dict["human_ssim"] = non_zero_mean(human_ssims) if compute_metrics else -1
    results_dict["vehicle_psnr"] = non_zero_mean(vehicle_psnrs) if compute_metrics else -1
    results_dict["vehicle_ssim"] = non_zero_mean(vehicle_ssims) if compute_metrics else -1
    results_dict["rgbs"] = rgbs
    results_dict["depths"] = depths
    results_dict["normals"] = normals
    if len(diffuse_lights)>0:
        results_dict['diffuse_light']=diffuse_lights
    if len(gt_normals) > 0:
        results_dict['gt_normals']=gt_normals
    if len(albedos) > 0:
        results_dict["rendered_albedos"] = albedos
    if len(rendered_roughness) > 0:
        results_dict["rendered_roughness"] = rendered_roughness
    if len(rendered_metallic) > 0:
        results_dict["rendered_metallic"] = rendered_metallic
    if len(rendered_reflectivity) > 0:
        results_dict["rendered_reflectivity"] = rendered_reflectivity
    if len(rendered_intensity) > 0:
        results_dict["rendered_intensity"] = rendered_intensity
    if len(gt_intensitys) > 0:
        results_dict["gt_intensity"] = gt_intensitys
    if len(rendered_sun_visibility) > 0:
        results_dict["rendered_sun_visibility"] = rendered_sun_visibility
    if len(rendered_dynamic_box_sun_visibility) > 0:
        results_dict["rendered_dynamic_box_sun_visibility"] = rendered_dynamic_box_sun_visibility
    if len(rendered_dynamic_box_contact_shadow) > 0:
        results_dict["rendered_dynamic_box_contact_shadow"] = rendered_dynamic_box_contact_shadow
    if len(rendered_dynamic_box_contact_shadow_raw) > 0:
        results_dict["rendered_dynamic_box_contact_shadow_raw"] = rendered_dynamic_box_contact_shadow_raw
    if len(rendered_dynamic_box_contact_apply_mask) > 0:
        results_dict["rendered_dynamic_box_contact_apply_mask"] = rendered_dynamic_box_contact_apply_mask
    if len(rendered_dynamic_opacity) > 0:
        results_dict["rendered_dynamic_opacity"] = rendered_dynamic_opacity
    if len(rendered_normals)>0:
        results_dict["rendered_normal"] = rendered_normals
    if len(pbrs)>0:
        results_dict["rendered_pbr"] = pbrs
    if len(pbrs_layered)>0:
        results_dict["rendered_pbr_layered"] = pbrs_layered
    if len(pbrs_background)>0:
        results_dict["rendered_pbr_background"] = pbrs_background
    if len(pbrs_dynamic)>0:
        results_dict["rendered_pbr_dynamic"] = pbrs_dynamic
    if len(rendered_pbr_dynamic_composite_mask)>0:
        results_dict["rendered_pbr_dynamic_composite_mask"] = rendered_pbr_dynamic_composite_mask
    results_dict["cam_names"] = cam_names
    results_dict["cam_ids"] = cam_ids
    if len(opacities) > 0:
        results_dict["opacities"] = opacities
    if len(gt_rgbs) > 0:
        results_dict["gt_rgbs"] = gt_rgbs
    if len(gt_albedos) > 0:
        results_dict["gt_albedos"] = gt_albedos
    if len(error_maps) > 0:
        results_dict["rgb_error_maps"] = error_maps
    if len(rgb_sky_blend) > 0:
        results_dict["rgb_sky_blend"] = rgb_sky_blend
    if len(rgb_sky) > 0:
        results_dict["rgb_sky"] = rgb_sky
    if len(sky_masks) > 0:
        results_dict["gt_sky_masks"] = sky_masks
    if len(lidar_on_images) > 0:
        results_dict["lidar_on_images"] = lidar_on_images
    if len(Background_rgbs) > 0:
        results_dict["Background_rgbs"] = Background_rgbs
    if len(RigidNodes_rgbs) > 0:
        results_dict["RigidNodes_rgbs"] = RigidNodes_rgbs
    if len(DeformableNodes_rgbs) > 0:
        results_dict["DeformableNodes_rgbs"] = DeformableNodes_rgbs
    if len(SMPLNodes_rgbs) > 0:
        results_dict["SMPLNodes_rgbs"] = SMPLNodes_rgbs
    if len(Dynamic_rgbs) > 0:
        results_dict["Dynamic_rgbs"] = Dynamic_rgbs
    if len(Background_depths) > 0:
        results_dict["Background_depths"] = Background_depths
    if len(RigidNodes_depths) > 0:
        results_dict["RigidNodes_depths"] = RigidNodes_depths
    if len(DeformableNodes_depths) > 0:
        results_dict["DeformableNodes_depths"] = DeformableNodes_depths
    if len(SMPLNodes_depths) > 0:
        results_dict["SMPLNodes_depths"] = SMPLNodes_depths
    if len(Dynamic_depths) > 0:
        results_dict["Dynamic_depths"] = Dynamic_depths
    if len(Background_opacities) > 0:
        results_dict["Background_opacities"] = Background_opacities
    if len(RigidNodes_opacities) > 0:
        results_dict["RigidNodes_opacities"] = RigidNodes_opacities
    if len(DeformableNodes_opacities) > 0:
        results_dict["DeformableNodes_opacities"] = DeformableNodes_opacities
    if len(SMPLNodes_opacities) > 0:
        results_dict["SMPLNodes_opacities"] = SMPLNodes_opacities
    if len(Dynamic_opacities) > 0:
        results_dict["Dynamic_opacities"] = Dynamic_opacities
    return results_dict


def save_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_seperate_video: bool = False,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):  
    if save_seperate_video:
        return_frame = save_seperate_videos(
            render_results,
            save_pth,
            layout,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    else:
        return_frame = save_concatenated_videos(
            render_results,
            save_pth,
            layout,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    return return_frame


def render_novel_views(trainer, render_data: list, save_path: str, fps: int = 30) -> None:
    """
    Perform rendering and save the result as a video.
    
    Args:
        trainer: Trainer object containing the rendering method
        render_data (list): List of dicts, each containing elements required for rendering a single frame
        save_path (str): Path to save the output video
        fps (int): Frames per second for the output video
    """
    trainer.set_eval()  
    
    writer = imageio.get_writer(save_path, mode='I', fps=fps)
    
    with torch.no_grad():
        for frame_data in render_data:
            # Move data to GPU
            for key, value in frame_data["cam_infos"].items():
                frame_data["cam_infos"][key] = value.cuda(non_blocking=True)
            for key, value in frame_data["image_infos"].items():
                frame_data["image_infos"][key] = value.cuda(non_blocking=True)
            
            # Perform rendering
            outputs = trainer(
                image_infos=frame_data["image_infos"],
                camera_infos=frame_data["cam_infos"],
                novel_view=True
            )
            
            # Extract RGB image and mask
            rgb = outputs["rgb"].cpu().numpy().clip(
                min=1.e-6, max=1-1.e-6
            )
            
            # Convert to uint8 and write to video
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            writer.append_data(rgb_uint8)
    
    writer.close()
    print(f"Video saved to {save_path}")


def save_concatenated_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if num_timestamps == 1:  # it's an image
        writer = imageio.get_writer(save_pth, mode="I")
        return_frame_id = 0
    else:
        return_frame_id = num_timestamps // 2
        writer = imageio.get_writer(save_pth, mode="I", fps=fps)
    for i in trange(num_timestamps, desc="saving video", dynamic_ncols=True):
        merged_list = []
        cam_names = render_results["cam_names"][i * num_cams : (i + 1) * num_cams]
        for key in keys:
            # skip if the key is not in render_results
            if "mask" in key:
                new_key = key.replace("mask", "opacities")
                if new_key not in render_results or len(render_results[new_key]) == 0:
                    continue
                frames = render_results[new_key][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            # convert to rgb if necessary
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif "mask" in key:
                frames = [
                    np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            elif "depth" in key:
                try:
                    opacities = render_results[key.replace("depths", "opacities")][
                        i * num_cams : (i + 1) * num_cams
                    ]
                except:
                    if "median" in key:
                        opacities = render_results[
                            key.replace("median_depths", "opacities")
                        ][i * num_cams : (i + 1) * num_cams]
                    else:
                        continue
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            tiled_img = layout(frames, cam_names)
            # frames = np.concatenate(frames, axis=1)
            merged_list.append(tiled_img)
        merged_frame = to8b(np.concatenate(merged_list, axis=0))
        if i == return_frame_id:
            return_frame = merged_frame
        writer.append_data(merged_frame)
    writer.close()
    if verbose:
        logger.info(f"saved video to {save_pth}")
    del render_results
    return {"concatenated_frame": return_frame}


def save_seperate_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    fps: int = 10,
    verbose: bool = False,
    save_images: bool = False,
):
    return_frame_id = num_timestamps // 2
    return_frame_dict = {}
    for key in keys:
        tmp_save_pth = save_pth.replace(".mp4", f"_{key}.mp4")
        tmp_save_pth = tmp_save_pth.replace(".png", f"_{key}.png")
        if num_timestamps == 1:  # it's an image
            writer = imageio.get_writer(tmp_save_pth, mode="I")
        else:
            writer = imageio.get_writer(tmp_save_pth, mode="I", fps=fps)
        if "mask" not in key:
            if key not in render_results or len(render_results[key]) == 0:
                continue
        for i in range(num_timestamps):
            cam_names = render_results["cam_names"][i * num_cams : (i + 1) * num_cams]
            # skip if the key is not in render_results
            if "mask" in key:
                new_key = key.replace("mask", "opacities")
                if new_key not in render_results or len(render_results[new_key]) == 0:
                    continue
                frames = render_results[new_key][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            # convert to rgb if necessary
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif "mask" in key:
                frames = [
                    np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            elif "depth" in key:
                try:
                    opacities = render_results[key.replace("depths", "opacities")][
                        i * num_cams : (i + 1) * num_cams
                    ]
                except:
                    if "median" in key:
                        opacities = render_results[
                            key.replace("median_depths", "opacities")
                        ][i * num_cams : (i + 1) * num_cams]
                    else:
                        continue
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            tiled_img = layout(frames, cam_names)
            if save_images:
                if i == 0:
                    os.makedirs(tmp_save_pth.replace(".mp4", ""), exist_ok=True)
                for j, frame in enumerate(frames):
                    imageio.imwrite(
                        tmp_save_pth.replace(".mp4", f"/{i:03d}_{j:03d}.png"),
                        to8b(frame),
                    )
            # frames = to8b(np.concatenate(frames, axis=1))
            frames = to8b(tiled_img)
            writer.append_data(frames)
            if i == return_frame_id:
                return_frame_dict[key] = frames
        # close the writer
        writer.close()
        del writer
        if verbose:
            logger.info(f"saved video to {tmp_save_pth}")
    del render_results
    return return_frame_dict
