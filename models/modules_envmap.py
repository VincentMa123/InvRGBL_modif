import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvironmentMap(nn.Module):
    """
    Learnable equirectangular environment map for image-based lighting.
    Uses pure PyTorch (no nvdiffrast required).
    
    Args:
        h: height of equirectangular map (default 32)
        w: width of equirectangular map (default 64, should be 2*h)
        min_roughness: minimum roughness for mip selection
        max_roughness: maximum roughness for mip selection
    """
    def __init__(self, h=32, w=64, min_roughness=0.08, max_roughness=0.5, class_name="EnvMap", **kwargs):
        super().__init__()
        self.class_prefix = class_name + "#"
        assert w == 2 * h, "Equirectangular width should be 2*height"
        self.h = h
        self.w = w
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        
        # Learnable base map [H, W, 3]
        base = torch.rand(h, w, 3, dtype=torch.float32) * 0.3 + 0.35
        self.base = nn.Parameter(base)
        self.build_mips()
    
    def build_mips(self):
        """Build Gaussian pyramid for prefiltered specular sampling."""
        self.specular = []
        current = self.base.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        self.specular.append(current.clone())
        
        # Build downsampled mips
        while min(current.shape[-2:]) > 1:
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
            self.specular.append(current.clone())
        
        # Diffuse: very blurry (Lambertian convolution approximation)
        self.diffuse = F.avg_pool2d(
            self.specular[-1], 
            kernel_size=self.specular[-1].shape[-2:], 
            stride=1
        )  # [1, 3, 1, 1]
        
        # Number of mips available
        self.num_mips = len(self.specular)
    
    def _directions_to_uv(self, directions):
        """
        Convert world directions to equirectangular UV coordinates.
        
        Args:
            directions: [N, 3] normalized world directions (x, y, z)
        Returns:
            uv: [N, 2] in range [-1, 1] for grid_sample
        """
        # directions: x=right, y=up, z=backward (OpenGL convention)
        # theta = atan2(x, -z)  # azimuth, range [-pi, pi]
        # phi = acos(y)         # elevation, range [0, pi]
        
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
        
        # Avoid atan2(0,0) singularity at poles (x=0, z=0) which gives NaN backward.
        # At poles azimuth is irrelevant for equirectangular sampling, so perturb slightly.
        pole_mask = (x.abs() < 1e-6) & (z.abs() < 1e-6)
        x_safe = torch.where(pole_mask, torch.ones_like(x) * 1e-6, x)
        z_safe = torch.where(pole_mask, torch.zeros_like(z), z)
        
        theta = torch.atan2(x_safe, -z_safe)  # [-pi, pi]
        # Clamp away from poles to avoid acos singularity (derivative -> -inf at y=±1)
        y_safe = y.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        phi = torch.acos(y_safe)  # [0, pi]
        
        # Map to [0, 1]
        u = theta / (2 * math.pi) + 0.5  # [0, 1]
        v = phi / math.pi  # [0, 1]
        
        # Map to [-1, 1] for grid_sample
        u = u * 2.0 - 1.0
        v = v * 2.0 - 1.0
        
        uv = torch.stack([u, v], dim=-1)
        return uv
    
    def sample(self, directions, roughness=None):
        """
        Sample environment map at given directions.
        
        Args:
            directions: [N, 3] normalized world directions
            roughness: [N, 1] or scalar. If provided, selects mip level.
        Returns:
            rgb: [N, 3]
        """
        if roughness is None:
            mip_level = 0
        else:
            if not isinstance(roughness, torch.Tensor):
                roughness = torch.tensor(roughness, device=directions.device)
            # Map roughness to mip level
            # roughness 0 -> mip 0 (sharp)
            # roughness 1 -> last mip (blurry)
            t = (roughness.clamp(self.min_roughness, 1.0) - self.min_roughness) \
                / (1.0 - self.min_roughness)
            mip_level = t * (self.num_mips - 1)
            mip_level = mip_level.long().clamp(0, self.num_mips - 1)
        
        uv = self._directions_to_uv(directions)  # [N, 2]
        
        if roughness is not None and roughness.numel() > 1:
            # Per-sample mip levels — need to sample individually
            # For simplicity, use the average mip level
            mip_level = mip_level.float().mean().long().item()
        else:
            mip_level = int(mip_level) if isinstance(mip_level, (int, float)) else mip_level.item()
        
        mip = self.specular[mip_level]  # [1, 3, H', W']
        
        # grid_sample expects [N, H, W, 2] uv and [1, C, H, W] image
        uv_batch = uv.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        
        rgb = F.grid_sample(
            mip,
            uv_batch,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [1, 3, 1, N]
        
        rgb = rgb.squeeze(0).squeeze(1).permute(1, 0)  # [N, 3]
        return rgb
    
    def sample_diffuse(self, directions):
        """
        Sample diffuse irradiance (blurriest mip = Lambertian approximation).
        
        Args:
            directions: [N, 3]
        Returns:
            rgb: [N, 3]
        """
        # Use the blurriest mip for diffuse (same as roughness=1.0)
        return self.sample(directions, roughness=1.0)
    
    def sample_specular(self, directions, roughness):
        """
        Sample prefiltered specular environment map.
        
        Args:
            directions: [N, 3] reflection directions
            roughness: [N, 1] or [1, 1]
        Returns:
            rgb: [N, 3]
        """
        return self.sample(directions, roughness)
    
    def get_param_groups(self):
        return {
            "EnvMap#all": self.parameters(),
        }
    
    def forward(self, directions):
        """Convenience: sample base map."""
        return self.sample(directions)
