import torch
from utils.graphics_utils import sample_incident_rays
import torch.nn.functional as F
import numpy as np

def srgb_to_linear(x):
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))

def linear_to_srgb(x):
    x = x.clamp_min(0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.pow(1.0 / 2.4) - 0.055)

def reinhard_tonemap(x):
    return x / (1.0 + x)

def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    half_vec = (L + V[:, None, :]) / 2.0
    # Guard NaN when L and V are exactly opposite (zero half-vector)
    zero_mask = half_vec.norm(dim=-1, keepdim=True) < 1e-6
    half_vec = torch.where(zero_mask, torch.tensor([0.0, 0.0, 1.0], device=half_vec.device, dtype=half_vec.dtype), half_vec)
    H = F.normalize(half_vec, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec



### USING ###
def rendering_equation_lidar(base_color, roughness, normals, viewdirs, view_dists):
    normals = normals.detach()

    # Ensure viewdirs has shape (N, 1, 3) for broadcasting
    if viewdirs.dim() == 2:
        viewdirs = viewdirs[:, None, :]  # (N, 1, 3)

    # cos(theta) = n · omega_o
    n_d_i = (normals[:, None, :] * viewdirs).sum(-1, keepdim=True).clamp(min=1e-6)
    cos_theta = n_d_i
    cos2_theta = cos_theta ** 2

    # Diffuse term: fd = rho_lidar / pi
    f_d = base_color[:, None, :] / np.pi

    # Specular term from paper (special case of Cook-Torrance with wi = wo)
    # fs = F0 * tau^2 * min(1, 2*cos^2(theta)) / (4*pi*cos^2(theta)*(cos^2(theta)*(tau^2-1)+1)^2)
    tau = roughness[:, None, :]
    tau2 = tau ** 2
    F0 = 0.04

    numerator = F0 * tau2 * torch.clamp(2 * cos2_theta, max=1.0)
    denominator = 4 * np.pi * cos2_theta * (cos2_theta * (tau2 - 1) + 1) ** 2
    f_s = numerator / denominator.clamp(min=1e-6)

    # Full LiDAR intensity: I = (fd + fs) * cos(theta)
    # NOTE: removed 1/d^2 term because GT Waymo intensity is already
    # range-corrected by sensor hardware / LUT normalization.
    pbr = ((f_d + f_s) * cos_theta).mean(dim=-2)
    return pbr



### USING ###
def rendering_equation(base_color, roughness, normals, viewdirs,
                              incidents=None, direct_light_env_light=None,
                              incident_dirs=None, incident_areas=None, visibility_precompute=None,sample_num=24,sun_visibility=None,xyz=None,step=None):
    
    normals = normals.detach()
    # if incident_dirs is None:
    #     incident_dirs, incident_areas = sample_incident_rays(normals, True, sample_num)
    with_sun = True
    if with_sun: #else:
        sun_visibility = visibility_precompute[:,0,:]
        sun_direction = incident_dirs[:,0,:]
        incident_areas = incident_areas[:,1:,:]
        visibility_precompute = visibility_precompute[:,1:,:]
        incident_dirs = incident_dirs[:,1:,:]
    else:        
        sun_direction = direct_light_env_light.get_sun_direction()
        sun_direction = sun_direction[None,...].repeat(incident_dirs.shape[0],1)


    deg = int(np.sqrt(incidents.shape[1]) - 1)
    

    global_incident_lights = direct_light_env_light.direct_light(incident_dirs,step) #)

    # Allow learned sky_intensity to affect shading instead of hardcoding a constant.
    # If you need the old hardcoded warm-gray fallback, uncomment the next line:
    # global_incident_lights = torch.ones_like(global_incident_lights) * torch.tensor([200, 200, 180]).to(device=normals.device) / 255 * 1.5
    local_incident_lights = 0 #eval_sh(deg, incidents.transpose(1, 2).view(-1, 1, 3, (deg + 1) ** 2), incident_dirs).clamp_min(0)
    incident_visibility = visibility_precompute
    # incident_visibility[incident_visibility>0.5] = 1
    # incident_visibility[incident_visibility<=0.5] = 0
    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = global_incident_lights + local_incident_lights  

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] / np.pi
    f_s = 0 #GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    

    specular = ((f_s) * transport).mean(dim=-2)
    pbr = ((f_d + f_s) * transport).mean(dim=-2)
    # Add small ambient term so fully BVH-occluded Gaussians aren't pitch black
    ambient = 0.02
    pbr = pbr + ambient
    diffuse_light = transport.mean(dim=-2) + ambient
    # Defensive: NaN/Inf from GGX zero half-vector or BVH cache corruption
    pbr = torch.nan_to_num(pbr, nan=ambient, posinf=1.0, neginf=0.0)
    diffuse_light = torch.nan_to_num(diffuse_light, nan=ambient, posinf=1.0, neginf=0.0)

    if with_sun and (sun_visibility is not None):
        sun_visibility = sun_visibility.detach()
        #sun_visibility = torch.where(sun_visibility < 0.95, torch.tensor(0.0), sun_visibility)
        intensity = direct_light_env_light.sun_intensity[None,...].repeat(incident_dirs.shape[0],1) #* 0.5
        #intensity = torch.ones_like(intensity) *torch.tensor([255, 178, 102]).to(device=intensity.device) * 3 / 255 #* 3
        sun_light = (sun_direction * normals).sum(dim=-1)[...,None] * intensity * 3#torch.ones_like(intensity) * 3 # * 10
        sun_light = sun_light.clip(0,10)
        f_d_ = (base_color / np.pi)
        f_s_ = GGX_specular(normals, viewdirs, sun_direction.unsqueeze(-2).detach(), roughness, fresnel=0.04).squeeze(-2)
        incident_sun_light = (f_d_ + f_s_) * sun_light * 0.3 
        pbr = pbr + incident_sun_light * sun_visibility 

    pbr = pbr 

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
        #"incident_visibility": incident_visibility,
        "diffuse_light":  diffuse_light, #specular
        "specular": specular,
        "incident_sun_light": sun_light ,
    }

    return pbr, extra_results

def compute_spotlight_contribution(spotlights, means, normals, albedos, roughness, viewdirs):
    """
    Additive spotlight contribution for inference-time relighting.
    
    Args:
        spotlights: list of dicts with keys:
            - "position": [3] torch.Tensor or list
            - "intensity": float
            - "color": [3] torch.Tensor or list (optional, default white)
            - "direction": [3] torch.Tensor or list (optional, for cone)
            - "cutoff_angle": float in radians (optional, default pi/6)
        means: [N, 3] Gaussian centers
        normals: [N, 3]
        albedos: [N, 3]
        roughness: [N, 1]
        viewdirs: [N, 3]
    
    Returns:
        [N, 3] RGB spotlight contribution
    """
    if spotlights is None or len(spotlights) == 0:
        return torch.zeros_like(albedos)
    
    total = torch.zeros_like(albedos)
    for sl in spotlights:
        pos = sl["position"]
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, device=means.device, dtype=torch.float32)
        pos = pos.to(means.device)
        
        # Vector from surface point to light
        pts2l = pos - means  # [N, 3]
        dist = torch.norm(pts2l, dim=-1, keepdim=True)
        pts2l = pts2l / (dist + 1e-6)
        
        # Distance falloff: inverse-linear with soft epsilon.
        # Inverse-square (1/dist^2) creates sharp hot-spots and ugly
        # interference stripes when multiple street-lamp pools overlap.
        # Inverse-linear (1/(dist+eps)) spreads light much more evenly.
        falloff = sl["intensity"] / (dist + 5.0)
        
        # Angular attenuation for spotlight cone
        if "direction" in sl and sl["direction"] is not None:
            direction = sl["direction"]
            if not torch.is_tensor(direction):
                direction = torch.tensor(direction, device=means.device, dtype=torch.float32)
            direction = F.normalize(direction, dim=-1)
            # cos of angle between -pts2l (vector TO light) and spotlight emission direction
            cos_theta = (-pts2l * direction).sum(dim=-1, keepdim=True)
            cutoff = sl.get("cutoff_angle", np.pi / 6.0)
            cos_cutoff = np.cos(cutoff)
            # Smooth step inside cone
            angular_atten = torch.clamp((cos_theta - cos_cutoff) / (1.0 - cos_cutoff + 1e-6), 0.0, 1.0)
        else:
            angular_atten = torch.ones_like(dist)
        
        # Diffuse BRDF term (without n·l)
        f_d = albedos / np.pi
        
        # Specular BRDF term
        f_s = GGX_specular(
            normals, viewdirs, pts2l.unsqueeze(1), roughness, fresnel=0.04
        ).squeeze(1)  # [N, 3]
        
        # n·l for cosine falloff
        n_dot_l = (normals * pts2l).sum(dim=-1, keepdim=True).clamp(min=0)
        
        # Light color
        color = sl.get("color", torch.ones(3, device=means.device))
        if not torch.is_tensor(color):
            color = torch.tensor(color, device=means.device, dtype=torch.float32)
        color = color.to(means.device)
        
        total = total + (f_d + f_s) * n_dot_l * falloff * angular_atten * color
    
    return total


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)



def compute_screen_space_ao(
    depth_map,
    normal_map=None,
    opacity_map=None,
    radius=5,
    strength=0.35,
    depth_bias=0.02,
):
    """Approximate ambient occlusion from local depth discontinuities."""
    if radius <= 0 or strength <= 0:
        return torch.ones_like(depth_map[..., :1])

    depth = depth_map[..., :1].permute(2, 0, 1).unsqueeze(0)
    finite = torch.isfinite(depth)
    valid = finite & (depth > 0)
    if opacity_map is not None:
        opacity = opacity_map[..., :1].permute(2, 0, 1).unsqueeze(0)
        valid = valid & (opacity > 1e-3)

    large_depth = torch.full_like(depth, 1e8)
    masked_depth = torch.where(valid, depth, large_depth)
    kernel_size = int(radius) * 2 + 1
    neg_masked_depth = F.pad(-masked_depth, (int(radius), int(radius), int(radius), int(radius)), mode="replicate")
    local_min_depth = -F.max_pool2d(
        neg_masked_depth,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
    )
    depth_delta = (depth - local_min_depth - float(depth_bias)).clamp(min=0.0)
    ao_raw = (depth_delta / depth.clamp(min=1e-3)).clamp(0.0, 1.0)

    if normal_map is not None:
        normal = F.normalize(normal_map.permute(2, 0, 1).unsqueeze(0), dim=1)
        padded_normal = F.pad(normal, (int(radius), int(radius), int(radius), int(radius)), mode="replicate")
        local_normal = F.avg_pool2d(
            padded_normal,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )
        local_normal = F.normalize(local_normal, dim=1)
        normal_variation = (1.0 - (normal * local_normal).sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
        ao_raw = ao_raw * (0.5 + 0.5 * normal_variation)

    ao = (1.0 - float(strength) * ao_raw).clamp(0.0, 1.0)
    ao = torch.where(valid, ao, torch.ones_like(ao))
    return ao.squeeze(0).permute(1, 2, 0)


def image_space_pbr(albedo_map, normal_map, roughness_map, metallic_map, sunvis_map,
                    viewdir_map, env_map, sun_dir, sun_intensity, spotlights=None,
                    depth_map=None, means_map=None, min_roughness=0.08,
                    reflectivity_map=None, reflectivity_f0_strength=0.35,
                    ao_map=None, specular_ao_strength=0.2):
    """
    Image-space PBR shading with environment map and analytic sun.
    
    This is the core deferred shading function that replaces per-Gaussian
    Monte-Carlo PBR with fast image-space IBL (image-based lighting).
    
    Args:
        albedo_map: [H, W, 3] rasterized albedo
        normal_map: [H, W, 3] rasterized normals (world space)
        roughness_map: [H, W, 1] rasterized roughness
        metallic_map: [H, W, 1] rasterized metallic
        sunvis_map: [H, W, 1] rasterized sun visibility (from BVH)
        viewdir_map: [H, W, 3] view directions per pixel (world space)
        env_map: EnvironmentMap instance
        sun_dir: [3] normalized sun direction
        sun_intensity: [3] linear sun RGB intensity
        spotlights: optional list of spotlights for inference
        depth_map: optional [H, W, 1] for spotlight distance computation
        means_map: optional [H, W, 3] world positions for spotlight computation
        ao_map: optional [H, W, 1] EnvMap visibility multiplier
    
    Returns:
        rgb: [H, W, 3] shaded image
    """
    import torch.nn.functional as F
    
    H, W = albedo_map.shape[:2]
    device = albedo_map.device
    
    # Flatten for batch processing
    N = H * W
    albedo = srgb_to_linear(albedo_map.reshape(N, 3))
    normal = F.normalize(normal_map.reshape(N, 3), dim=-1)
    roughness = roughness_map.reshape(N, 1).clamp(min_roughness, 1.0)
    metallic = metallic_map.reshape(N, 1).clamp(0, 1)
    sunvis = sunvis_map.reshape(N, 1)
    viewdir = F.normalize(viewdir_map.reshape(N, 3), dim=-1)
    
    # ---- Environment lighting (IBL) ----
    
    # Diffuse: sample env map at normal direction
    env_diffuse = env_map.sample_diffuse(normal)  # [N, 3]
    
    # Specular: sample env map at reflection direction with roughness-based blur
    NoV = (normal * viewdir).sum(-1, keepdim=True).clamp(min=0)
    reflection = 2 * NoV * normal - viewdir
    reflection = F.normalize(reflection, dim=-1)
    env_specular = env_map.sample_specular(reflection, roughness)  # [N, 3]
    if ao_map is not None:
        ao = ao_map.reshape(N, 1).clamp(0.0, 1.0)
        env_diffuse = env_diffuse * ao
        specular_ao = 1.0 - float(specular_ao_strength) * (1.0 - ao)
        env_specular = env_specular * specular_ao.clamp(0.0, 1.0)
    
    # Metallic Fresnel: F0 = (1-metallic)*dielectric_F0 + albedo*metallic.
    # LiDAR reflectivity is a near-infrared material cue, not visible RGB color.
    # Use it only as a neutral scalar prior for dielectric specular strength.
    dielectric_F0 = torch.full_like(metallic, 0.04)
    if reflectivity_map is not None:
        reflectivity = reflectivity_map.reshape(N, 1).clamp(0.0, 1.0)
        reflectivity_f0 = 0.02 + 0.06 * reflectivity
        strength = float(reflectivity_f0_strength)
        dielectric_F0 = dielectric_F0 + strength * (reflectivity_f0 - dielectric_F0)
    F0 = (1.0 - metallic) * dielectric_F0 + albedo * metallic
    fresnel = F0 + (1.0 - F0) * (1.0 - NoV).pow(5)
    
    # Combine: diffuse gets albedo tint, specular gets fresnel
    diffuse = albedo * env_diffuse * (1.0 - metallic)
    specular = env_specular * fresnel
    
    # ---- Analytic sun (with BVH shadows) ----
    
    sun_dir = F.normalize(sun_dir, dim=-1)
    NoL = (normal * sun_dir).sum(-1, keepdim=True).clamp(min=0)
    sun_light = NoL * F.softplus(sun_intensity.to(device)) * 3
    sun_light = sun_light.clip(0, 10)
    
    # Sun diffuse
    sun_diffuse = albedo / np.pi * sun_light * sunvis
    
    # Sun specular (simplified GGX in image space)
    sun_dir_expanded = sun_dir[None, None, :].expand(N, 1, 3)
    sun_specular_brdf = GGX_specular(
        normal, viewdir, sun_dir_expanded, roughness, fresnel=fresnel[:, None, :]
    ).squeeze(1)  # [N, 3]
    # For metallic, tint specular by F0 (approximation)
    sun_specular = sun_specular_brdf * sun_light * (F0 * 0.5 + 0.5) * sunvis * 0.3
    
    # ---- Spotlights (inference only) ----
    spotlight_contrib = 0
    if spotlights is not None and len(spotlights) > 0 and means_map is not None:
        means_flat = means_map.reshape(N, 3)
        spotlight_contrib = compute_spotlight_contribution(
            spotlights, means_flat, normal, albedo, roughness, viewdir
        )
    
    # ---- Combine all contributions ----
    pbr = diffuse + specular + sun_diffuse + sun_specular + spotlight_contrib
    pbr = linear_to_srgb(reinhard_tonemap(pbr))
    
    # Defensive: catch NaN/Inf early in image-space PBR before they corrupt means
    if torch.isnan(pbr).any() or torch.isinf(pbr).any():
        bad = torch.isnan(pbr) | torch.isinf(pbr)
        raise ValueError(
            f"NaN/Inf detected in image_space_pbr output at {bad.sum().item()} pixels. "
            f"diffuse nan={torch.isnan(diffuse).any()}, specular nan={torch.isnan(specular).any()}, "
            f"sun_diffuse nan={torch.isnan(sun_diffuse).any()}, sun_specular nan={torch.isnan(sun_specular).any()}"
        )
    
    return pbr.reshape(H, W, 3)
