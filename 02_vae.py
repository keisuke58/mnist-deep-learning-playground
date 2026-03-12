"""
Step 2: VAE (Variational Autoencoder) — 手書き数字を生成する
潜在空間を歩くと数字がモーフィングする
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)

LATENT_DIM = 2  # 2次元にして可視化しやすく


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)  # 14x14
        self.enc_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 7x7
        self.fc_mu = nn.Linear(64 * 7 * 7, LATENT_DIM)
        self.fc_logvar = nn.Linear(64 * 7 * 7, LATENT_DIM)

        # Decoder
        self.fc_dec = nn.Linear(LATENT_DIM, 64 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.view(-1, 64 * 7 * 7)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = F.relu(self.fc_dec(z)).view(-1, 64, 7, 7)
        h = F.relu(self.dec_conv1(h))
        return torch.sigmoid(self.dec_conv2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar):
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("=== Step 2: VAE Training ===")
for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        recon, mu, logvar = model(data)
        loss = vae_loss(recon, data, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_ds)
    print(f"Epoch {epoch:2d} | Loss: {avg_loss:.2f}")

# Generate: 潜在空間のグリッドから生成
print("\nGenerating latent space grid...")
import numpy as np
n = 20
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)

model.eval()
with torch.no_grad():
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
            digit = model.decode(z).cpu().numpy()[0, 0]
            figure[i * digit_size:(i + 1) * digit_size,
                   j * digit_size:(j + 1) * digit_size] = digit

from PIL import Image
img = Image.fromarray((figure * 255).astype(np.uint8))
img.save("outputs/vae_latent_space.png")
print("Saved: outputs/vae_latent_space.png")

# Generate random samples
z = torch.randn(64, LATENT_DIM).to(device)
with torch.no_grad():
    samples = model.decode(z)
save_image(samples, "outputs/vae_random_samples.png", nrow=8)
print("Saved: outputs/vae_random_samples.png")

# Morphing: 0 → 9 への補間
print("\nMorphing between digits...")
test_ds = datasets.MNIST("data", train=False, transform=transform)
# Find a 0 and a 9
digit_0 = next(x for x, y in test_ds if y == 0).unsqueeze(0).to(device)
digit_9 = next(x for x, y in test_ds if y == 9).unsqueeze(0).to(device)
with torch.no_grad():
    mu_0, _ = model.encode(digit_0)
    mu_9, _ = model.encode(digit_9)
    morphs = []
    for alpha in np.linspace(0, 1, 16):
        z = mu_0 * (1 - alpha) + mu_9 * alpha
        morphs.append(model.decode(z))
    morphs = torch.cat(morphs)
save_image(morphs, "outputs/vae_morph_0_to_9.png", nrow=16)
print("Saved: outputs/vae_morph_0_to_9.png")

torch.save(model.state_dict(), "vae.pth")
