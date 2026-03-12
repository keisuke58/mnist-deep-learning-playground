"""
Step 3: DCGAN — 手書き数字を生成 (敵対的生成)
Generator vs Discriminator のバトルで学習
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

LATENT_DIM = 100
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
            # z -> 256 x 7 x 7
            nn.ConvTranspose2d(LATENT_DIM, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # -> 128 x 14 x 14
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # -> 1 x 28 x 28
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(-1, LATENT_DIM, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


G = Generator().to(device)
D = Discriminator().to(device)
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, LATENT_DIM, device=device)

print("=== Step 3: DCGAN Training ===")
for epoch in range(1, 51):
    g_loss_sum, d_loss_sum = 0, 0
    for real, _ in train_loader:
        real = real.to(device)
        bs = real.size(0)
        real_label = torch.ones(bs, device=device)
        fake_label = torch.zeros(bs, device=device)

        # Train Discriminator
        D.zero_grad()
        d_real = D(real)
        loss_real = criterion(d_real, real_label)
        z = torch.randn(bs, LATENT_DIM, device=device)
        fake = G(z)
        d_fake = D(fake.detach())
        loss_fake = criterion(d_fake, fake_label)
        d_loss = loss_real + loss_fake
        d_loss.backward()
        opt_D.step()

        # Train Generator
        G.zero_grad()
        d_fake = D(fake)
        g_loss = criterion(d_fake, real_label)
        g_loss.backward()
        opt_G.step()

        g_loss_sum += g_loss.item()
        d_loss_sum += d_loss.item()

    n = len(train_loader)
    print(f"Epoch {epoch:2d} | D_loss: {d_loss_sum/n:.4f} | G_loss: {g_loss_sum/n:.4f}")

    if epoch % 10 == 0:
        with torch.no_grad():
            samples = G(fixed_noise)
        save_image(samples, f"outputs/gan_epoch{epoch:03d}.png", nrow=8, normalize=True)
        print(f"  -> Saved: outputs/gan_epoch{epoch:03d}.png")

torch.save(G.state_dict(), "gan_generator.pth")
print("Saved: gan_generator.pth")
