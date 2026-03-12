"""
Step 3: DCGAN — 手書き数字を生成 (敵対的生成)
Deeper architecture, spectral norm, 200 epochs for high quality
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

LATENT_DIM = 128
BATCH_SIZE = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # z -> 512 x 7 x 7
            nn.ConvTranspose2d(LATENT_DIM, 512, 7, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # -> 256 x 14 x 14
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # -> 128 x 28 x 28
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # -> 1 x 28 x 28
            nn.Conv2d(128, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(-1, LATENT_DIM, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, x):
        return self.net(x).view(-1)


G = Generator().to(device)
D = Discriminator().to(device)
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, LATENT_DIM, device=device)

print("=== Step 3: DCGAN Training (200 epochs) ===")
print(f"G params: {sum(p.numel() for p in G.parameters()):,}")
print(f"D params: {sum(p.numel() for p in D.parameters()):,}")

for epoch in range(1, 201):
    g_loss_sum, d_loss_sum = 0, 0
    for real, _ in train_loader:
        real = real.to(device)
        bs = real.size(0)

        # Train Discriminator (WGAN-like with hinge loss)
        D.zero_grad()
        d_real = D(real)
        z = torch.randn(bs, LATENT_DIM, device=device)
        fake = G(z)
        d_fake = D(fake.detach())
        d_loss = torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()
        d_loss.backward()
        opt_D.step()

        # Train Generator
        G.zero_grad()
        d_fake = D(fake)
        g_loss = -d_fake.mean()
        g_loss.backward()
        opt_G.step()

        g_loss_sum += g_loss.item()
        d_loss_sum += d_loss.item()

    n = len(train_loader)
    if epoch % 10 == 0 or epoch <= 5:
        print(f"Epoch {epoch:3d} | D_loss: {d_loss_sum/n:.4f} | G_loss: {g_loss_sum/n:.4f}")

    if epoch % 50 == 0:
        with torch.no_grad():
            samples = G(fixed_noise)
        save_image(samples, f"outputs/gan_epoch{epoch:03d}.png", nrow=8, normalize=True)
        print(f"  -> Saved: outputs/gan_epoch{epoch:03d}.png")

# Final generation
with torch.no_grad():
    samples = G(fixed_noise)
save_image(samples, "outputs/gan_final.png", nrow=8, normalize=True)
print("Saved: outputs/gan_final.png")

torch.save(G.state_dict(), "models/gan_generator.pth")
torch.save(D.state_dict(), "models/gan_discriminator.pth")
print("Saved: models/gan_generator.pth, models/gan_discriminator.pth")
