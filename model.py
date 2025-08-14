import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Utils ---
def _groups_ok(n_groups, n_channels):
    # make sure groups divide channels
    return max(1, min(n_groups, n_channels // max(1, (n_channels // n_groups))))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# --- Time Embedding ---
class TimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding + small MLP to `time_dim`.
    Expects t: (B,) or (B,1) float or long (will be cast to float).
    """
    def __init__(self, time_dim: int, max_period: int = 10_000):
        super().__init__()
        self.time_dim = time_dim
        self.max_period = max_period
        self.act = Swish()
        self.fc1 = nn.Linear(time_dim, time_dim * 2)
        self.fc2 = nn.Linear(time_dim * 2, time_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.size(1) == 1:
            t = t.squeeze(1)
        t = t.float()
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=t.device).float() / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(1) < self.time_dim:  # odd dim guard
            emb = F.pad(emb, (0, self.time_dim - emb.size(1)))
        emb = self.act(self.fc1(emb))
        emb = self.fc2(emb)
        return emb  # (B, time_dim)


# --- Core Blocks ---
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, n_groups=32):
        super().__init__()
        g1 = _groups_ok(n_groups, in_ch)
        g2 = _groups_ok(n_groups, out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Sequential(Swish(), nn.Linear(time_dim, out_ch))

        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention on (H*W) tokens.
    """
    def __init__(self, channels, n_heads=4, n_groups=32):
        super().__init__()
        assert channels % n_heads == 0, "channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

        self.norm = nn.GroupNorm(_groups_ok(n_groups, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_n = self.norm(x)

        q = self.q(x_n)
        k = self.k(x_n)
        v = self.v(x_n)

        # reshape -> (B, heads, HW, head_dim)
        def to_heads(t):
            t = t.view(b, self.n_heads, self.head_dim, h * w)
            return t.transpose(2, 3)  # (B, heads, HW, head_dim)

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        # attention: (B, heads, HW, HW)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        out = self.proj(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x, _t):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, _t):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# --- The EBM U-Net ---
class EBM_Unet(nn.Module):
    """
    Energy-Based U-Net.
    - Produces scalar energy per image: (B,)
    - `attn_res` lists the spatial sizes (e.g. 16, 32) where attention is inserted.
    - `input_size` controls where attention modules are placed during construction.
    """
    def __init__(
        self,
        image_channels=3,
        n_channels=128,
        ch_mults=(1, 2, 2, 2),
        n_blocks=2,
        attn_res=(16,),
        input_size=32,
        n_heads=4,
        n_groups=32,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.n_channels = n_channels
        self.ch_mults = tuple(ch_mults)
        self.n_blocks = n_blocks
        self.attn_res = set(attn_res)
        self.input_size = input_size
        self.n_heads = n_heads
        self.n_groups = n_groups

        # time embedding
        self.time_dim = n_channels * 4
        self.time_embed = TimeEmbedding(self.time_dim)

        # input
        self.conv_in = nn.Conv2d(image_channels, n_channels, 3, padding=1)

        # --- Down path ---
        self.down = nn.ModuleList()
        self.skip_channels = []  # record channels after each ResBlock (for up path sizing)
        cur_ch = n_channels
        cur_res = input_size

        for i, mult in enumerate(self.ch_mults):
            out_ch = n_channels * mult
            for _ in range(self.n_blocks):
                rb = ResidualBlock(cur_ch, out_ch, self.time_dim, n_groups)
                self.down.append(rb)
                cur_ch = out_ch
                self.skip_channels.append(cur_ch)           # track for up path sizing
                if cur_res in self.attn_res:
                    self.down.append(AttentionBlock(cur_ch, self.n_heads, n_groups))
            if i != len(self.ch_mults) - 1:
                self.down.append(Downsample(cur_ch))
                cur_res //= 2

        # --- Middle ---
        self.mid = nn.ModuleList([
            ResidualBlock(cur_ch, cur_ch, self.time_dim, n_groups),
            AttentionBlock(cur_ch, self.n_heads, n_groups),
            ResidualBlock(cur_ch, cur_ch, self.time_dim, n_groups),
        ])

        # --- Up path ---
        # We'll mirror the number of ResBlocks from the down path and consume skip channels in reverse.
        self.up = nn.ModuleList()
        skip_iter = list(self.skip_channels)
        cur_res_up = cur_res

        for i, mult in reversed(list(enumerate(self.ch_mults))):
            out_ch = n_channels * mult
            for _ in range(self.n_blocks):
                skip_ch = skip_iter.pop()
                rb = ResidualBlock(cur_ch + skip_ch, out_ch, self.time_dim, n_groups)
                self.up.append(rb)
                cur_ch = out_ch
                if cur_res_up in self.attn_res:
                    self.up.append(AttentionBlock(cur_ch, self.n_heads, n_groups))
            if i != 0:
                self.up.append(Upsample(cur_ch))
                cur_res_up *= 2

        # --- Output head (scalar energy) ---
        self.norm_out = nn.GroupNorm(_groups_ok(n_groups, cur_ch), cur_ch)
        self.act_out = Swish()
        self.energy_head = nn.Sequential(
            nn.Linear(cur_ch, cur_ch),
            Swish(),
            nn.Linear(cur_ch, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        t: (B,) or (B,1) timestep/noise level
        return: (B,) energy
        """
        B = x.size(0)
        t_emb = self.time_embed(t)

        h = self.conv_in(x)

        # Down path + collect skips
        skips = []
        for m in self.down:
            if isinstance(m, ResidualBlock):
                h = m(h, t_emb)
                skips.append(h)
            elif isinstance(m, AttentionBlock):
                h = m(h)
            elif isinstance(m, Downsample):
                h = m(h, t_emb)
            else:
                raise RuntimeError(f"Unexpected module in down path: {type(m)}")

        # Middle
        h = self.mid[0](h, t_emb)
        h = self.mid[1](h)
        h = self.mid[2](h, t_emb)

        # Up path (consume skips)
        for m in self.up:
            if isinstance(m, ResidualBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = m(h, t_emb)
            elif isinstance(m, AttentionBlock):
                h = m(h)
            elif isinstance(m, Upsample):
                h = m(h, t_emb)
            else:
                raise RuntimeError(f"Unexpected module in up path: {type(m)}")

        # Global pool -> MLP -> scalar
        h = self.act_out(self.norm_out(h))
        h = h.mean(dim=(2, 3))            # (B, C)
        energy = self.energy_head(h).squeeze(-1)  # (B,)
        return energy
