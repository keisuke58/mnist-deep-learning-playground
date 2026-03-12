"""
Step 6: Adversarial Attack — 99%精度のCNNを騙す
人間には見えない微小ノイズで誤分類させる
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)


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


# First train a model
print("=== Step 6: Adversarial Attacks ===")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("data", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1)

model = SimpleCNN().to(device)

if os.path.exists("basic_cnn.pth"):
    model.load_state_dict(torch.load("basic_cnn.pth", map_location=device))
    print("Loaded pre-trained model")
else:
    print("Training model first...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(data), target).backward()
            optimizer.step()
    print("Training done")


def fgsm_attack(model, image, label, epsilon):
    """Fast Gradient Sign Method"""
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    perturbed = image + epsilon * image.grad.sign()
    return perturbed.detach()


def pgd_attack(model, image, label, epsilon=0.3, alpha=0.01, steps=40):
    """Projected Gradient Descent — stronger attack"""
    perturbed = image.clone().detach()
    for _ in range(steps):
        perturbed.requires_grad = True
        output = model(perturbed)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        adv = perturbed + alpha * perturbed.grad.sign()
        # Project back to epsilon-ball
        delta = torch.clamp(adv - image, -epsilon, epsilon)
        perturbed = (image + delta).detach()
    return perturbed


# Test attacks at different epsilon values
model.eval()
print("\n--- FGSM Attack ---")
for epsilon in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    correct = 0
    total = 0
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        adv = fgsm_attack(model, data, target, epsilon)
        pred = model(adv).argmax(dim=1)
        if pred.item() == target.item():
            correct += 1
        elif len(adv_examples) < 5:
            adv_examples.append((data, adv, target.item(), pred.item()))
        total += 1
        if total >= 1000:
            break
    acc = correct / total
    print(f"  eps={epsilon:.2f} | Acc: {acc*100:.1f}%")

    if epsilon == 0.2 and adv_examples:
        imgs = []
        for orig, adv, true_l, pred_l in adv_examples:
            imgs.extend([orig.cpu(), adv.cpu(), (adv - orig).cpu() * 5])
        imgs = torch.cat(imgs)
        save_image(imgs, "outputs/adversarial_fgsm.png", nrow=3, normalize=True)
        print("  -> Saved: outputs/adversarial_fgsm.png (orig | adversarial | perturbation x5)")

# PGD Attack
print("\n--- PGD Attack (stronger) ---")
correct = 0
total = 0
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    adv = pgd_attack(model, data, target, epsilon=0.3)
    pred = model(adv).argmax(dim=1)
    if pred.item() == target.item():
        correct += 1
    total += 1
    if total >= 500:
        break
print(f"  PGD (eps=0.3, 40 steps) | Acc: {correct/total*100:.1f}%")

# Targeted attack: make all digits look like "3"
print("\n--- Targeted Attack: everything -> 3 ---")
target_label = torch.tensor([3], device=device)
successes = []
for data, true_label in test_loader:
    if true_label.item() == 3:
        continue
    data = data.to(device)
    perturbed = data.clone().detach()
    for _ in range(100):
        perturbed.requires_grad = True
        output = model(perturbed)
        loss = F.cross_entropy(output, target_label)
        model.zero_grad()
        loss.backward()
        perturbed = (perturbed - 0.01 * perturbed.grad.sign()).detach()
        delta = torch.clamp(perturbed - data, -0.3, 0.3)
        perturbed = data + delta

    pred = model(perturbed).argmax(dim=1)
    if pred.item() == 3:
        successes.append((data.cpu(), perturbed.cpu(), true_label.item()))
    if len(successes) >= 8:
        break

if successes:
    imgs = []
    for orig, adv, true_l in successes:
        imgs.extend([orig, adv])
    imgs = torch.cat(imgs)
    save_image(imgs, "outputs/targeted_attack_to_3.png", nrow=2, normalize=True)
    print(f"  Fooled {len(successes)} digits into being classified as 3")
    print("  -> Saved: outputs/targeted_attack_to_3.png (orig | fooled)")
