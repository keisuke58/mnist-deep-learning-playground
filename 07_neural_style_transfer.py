"""
Step 7: Neural Style Transfer on MNIST
手書き数字にアーティスティックなスタイルを適用
+ ニューラルネットの「見ているもの」を可視化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

# ===== Part A: Feature Visualization (DeepDream風) =====
print("=== Step 7A: Feature Visualization ===")
print("What does the CNN 'see' in each digit?")


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


model = SimpleCNN().to(device)

# Train quickly if no checkpoint
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

if os.path.exists("basic_cnn.pth"):
    model.load_state_dict(torch.load("basic_cnn.pth", map_location=device))
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(data), target).backward()
            optimizer.step()

model.eval()

# Generate "ideal" digit for each class — what does the network think a perfect 0-9 looks like?
print("Generating ideal digits (gradient ascent on class scores)...")
ideal_digits = []
for digit in range(10):
    img = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([img], lr=0.05)
    for step in range(500):
        optimizer.zero_grad()
        output = model(img)
        loss = -output[0, digit] + 0.01 * img.pow(2).sum()  # maximize class score + regularize
        loss.backward()
        optimizer.step()
    ideal_digits.append(img.detach().cpu())
    print(f"  Digit {digit}: score = {model(img)[0, digit].item():.2f}")

ideal_all = torch.cat(ideal_digits)
save_image(ideal_all, "outputs/ideal_digits.png", nrow=5, normalize=True)
print("Saved: outputs/ideal_digits.png")

# ===== Part B: Filter Visualization =====
print("\n=== Step 7B: Conv Filter Visualization ===")
filters = model.conv1.weight.data.cpu()  # 32 x 1 x 3 x 3
save_image(filters, "outputs/conv1_filters.png", nrow=8, normalize=True, padding=1)
print("Saved: outputs/conv1_filters.png (32 learned 3x3 filters)")

# ===== Part C: Activation Maps =====
print("\n=== Step 7C: Activation Maps ===")
test_ds = datasets.MNIST("data", train=False, transform=transforms.ToTensor())
sample_img = test_ds[0][0].unsqueeze(0).to(device)  # a "7"

# Hook to capture activations
activations = {}
def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

model.conv1.register_forward_hook(hook_fn("conv1"))
model.conv2.register_forward_hook(hook_fn("conv2"))

with torch.no_grad():
    _ = model(transforms.Normalize((0.1307,), (0.3081,))(sample_img))

# Save conv1 activations
act1 = activations["conv1"]  # 1 x 32 x 14 x 14
act_imgs = act1[0].unsqueeze(1)  # 32 x 1 x 14 x 14
save_image(act_imgs, "outputs/activations_conv1.png", nrow=8, normalize=True, padding=1)
print("Saved: outputs/activations_conv1.png")

act2 = activations["conv2"]  # 1 x 64 x 7 x 7
act_imgs2 = act2[0].unsqueeze(1)  # 64 x 1 x 7 x 7
save_image(act_imgs2, "outputs/activations_conv2.png", nrow=8, normalize=True, padding=1)
print("Saved: outputs/activations_conv2.png")

# ===== Part D: t-SNE of learned features =====
print("\n=== Step 7D: t-SNE Feature Embedding ===")
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model.eval()
features_list = []
labels_list = []

# Get features from fc1
fc1_hook = {}
def fc1_hook_fn(module, input, output):
    fc1_hook["feat"] = output.detach()
model.fc1.register_forward_hook(fc1_hook_fn)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, transform=transform),
    batch_size=256
)

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        _ = model(data)
        features_list.append(fc1_hook["feat"].cpu().numpy())
        labels_list.append(target.numpy())

features = np.concatenate(features_list)
labels = np.concatenate(labels_list)

print(f"Running t-SNE on {len(features)} samples...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embedding = tsne.fit_transform(features)

plt.figure(figsize=(10, 10))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=1, alpha=0.6)
plt.colorbar(scatter, ticks=range(10))
plt.title("t-SNE of CNN Features (fc1 layer)")
plt.savefig("outputs/tsne_features.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/tsne_features.png")
