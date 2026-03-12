"""
Step 4: Diffusion Model (DDPM) — 今一番ホットな生成モデル
ノイズから徐々にデノイズして数字を生成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

T = 1000  # diffusion steps
BATCH_SIZE = 128

# Beta schedule
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class UNet(nn.Module):
    """Simplified U-Net for 28x28"""
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU())

        # Time projection
        self.time_proj = nn.Linear(128, 128)

        # Bottleneck
        self.bot = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU())

        # Decoder
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.out = nn.Conv2d(128, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.float())

        # Encode
        h1 = self.enc1(x)       # 64 x 28 x 28
        h2 = self.enc2(h1)      # 128 x 14 x 14
        h3 = self.enc3(h2)      # 128 x 7 x 7

        # Add time
        h3 = h3 + self.time_proj(t_emb)[:, :, None, None]

        # Bottleneck
        h = self.bot(h3)        # 128 x 7 x 7

        # Decode with skip connections
        h = self.dec3(torch.cat([h, h3], dim=1))   # 128 x 14 x 14
        h = self.dec2(torch.cat([h, h2], dim=1))   # 64 x 28 x 28
        h = self.out(torch.cat([h, h1], dim=1))     # 1 x 28 x 28
        return h


def q_sample(x0, t, noise=None):
    """Add noise to x0 at timestep t"""
    if noise is None:
        noise = torch.randn_like(x0)
    ab = alpha_bar[t][:, None, None, None]
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise


@torch.no_grad()
def p_sample(model, x, t_idx):
    """Denoise one step"""
    t = torch.full((x.size(0),), t_idx, device=device, dtype=torch.long)
    pred_noise = model(x, t)
    b = beta[t_idx]
    a = alpha[t_idx]
    ab = alpha_bar[t_idx]
    x = (1 / torch.sqrt(a)) * (x - (b / torch.sqrt(1 - ab)) * pred_noise)
    if t_idx > 0:
        x += torch.sqrt(b) * torch.randn_like(x)
    return x


@torch.no_grad()
def generate(model, n=64):
    """Generate images from noise"""
    x = torch.randn(n, 1, 28, 28, device=device)
    for t in reversed(range(T)):
        x = p_sample(model, x, t)
    return x.clamp(-1, 1)


model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

print("=== Step 4: Diffusion Model Training ===")
for epoch in range(1, 31):
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        t = torch.randint(0, T, (data.size(0),), device=device)
        noisy, noise = q_sample(data, t)
        pred = model(noisy, t)
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(train_loader)
    print(f"Epoch {epoch:2d} | Loss: {avg:.5f}")

    if epoch % 10 == 0:
        samples = generate(model, 64)
        save_image(samples, f"outputs/diffusion_epoch{epoch:03d}.png", nrow=8, normalize=True, value_range=(-1, 1))
        print(f"  -> Saved: outputs/diffusion_epoch{epoch:03d}.png")

torch.save(model.state_dict(), "diffusion.pth")
print("Saved: diffusion.pth")
