"""Generate extra visualizations for README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import os

BG = "#0d1117"
TEXT = "white"
SUB = "#8b949e"
ACCENT = "#58a6ff"

# === 1. RL Learning Curve ===
print("1. RL reward curve...")
rewards = {
    500: -2.660, 1000: -0.039, 1500: -0.028, 2000: -0.002,
    2500: -0.001, 3000: -0.004, 3500: -2.549, 4000: 0.587,
    4500: 6.756, 5000: 1.481
}
eps = list(rewards.keys())
rews = list(rewards.values())

fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
ax.set_facecolor(BG)
ax.plot(eps, rews, color=ACCENT, linewidth=2.5, marker="o", markersize=6)
ax.fill_between(eps, rews, alpha=0.15, color=ACCENT)
ax.axhline(y=0, color="#30363d", linewidth=0.8, linestyle="--")
ax.scatter([4500], [6.756], color="#f0883e", s=120, zorder=5, label="Best: 6.756")
ax.set_xlabel("Episode", color=SUB, fontsize=12)
ax.set_ylabel("Avg Reward", color=SUB, fontsize=12)
ax.set_title("RL Digit Writer — Training Curve", color=TEXT, fontsize=16, fontweight="bold")
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=TEXT, fontsize=11)
ax.tick_params(colors=SUB)
for spine in ax.spines.values():
    spine.set_color("#30363d")
ax.grid(axis="y", color="#21262d", linewidth=0.5)
plt.tight_layout()
plt.savefig("outputs/rl_reward_curve.png", dpi=150, facecolor=BG, bbox_inches="tight")
plt.close()
print("  Saved: outputs/rl_reward_curve.png")

# === 2. RL Detailed Progression (more steps) ===
print("2. RL detailed progression...")
steps = [
    ("outputs/rl_writer_ep0100.png", "Ep 100"),
    ("outputs/rl_writer_ep0500.png", "Ep 500"),
    ("outputs/rl_writer_ep01000.png", "Ep 1000"),
    ("outputs/rl_writer_ep02000.png", "Ep 2000"),
    ("outputs/rl_writer_ep03000.png", "Ep 3000"),
    ("outputs/rl_writer_ep04000.png", "Ep 4000"),
    ("outputs/rl_writer_ep04500.png", "Ep 4500\n(best)"),
    ("outputs/rl_writer_ep05000.png", "Ep 5000"),
]
steps = [(p, l) for p, l in steps if os.path.exists(p)]

fig, axes = plt.subplots(1, len(steps), figsize=(3.5 * len(steps), 3.5), facecolor=BG)
fig.suptitle("RL Digit Writer — Drawing '7' Over Training",
             fontsize=18, color=TEXT, fontweight="bold", y=1.02)

for i, (path, label) in enumerate(steps):
    ax = axes[i]
    ax.set_facecolor(BG)
    img = Image.open(path)
    ax.imshow(np.array(img))
    color = "#f0883e" if "best" in label else TEXT
    ax.set_title(label, color=color, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#30363d")

plt.tight_layout()
plt.savefig("outputs/rl_detailed_progression.png", dpi=150, facecolor=BG, bbox_inches="tight")
plt.close()
print("  Saved: outputs/rl_detailed_progression.png")

# === 3. GAN Training Progression ===
print("3. GAN progression...")
gan_files = [
    ("outputs/gan_epoch010.png", "Epoch 10"),
    ("outputs/gan_epoch020.png", "Epoch 20"),
    ("outputs/gan_epoch050.png", "Epoch 50"),
    ("outputs/gan_epoch100.png", "Epoch 100"),
    ("outputs/gan_epoch150.png", "Epoch 150"),
    ("outputs/gan_epoch200.png", "Epoch 200"),
]
gan_files = [(p, l) for p, l in gan_files if os.path.exists(p)]

fig, axes = plt.subplots(1, len(gan_files), figsize=(4 * len(gan_files), 4), facecolor=BG)
fig.suptitle("DCGAN — Generation Quality Over Training",
             fontsize=18, color=TEXT, fontweight="bold", y=1.02)

for i, (path, label) in enumerate(gan_files):
    ax = axes[i]
    ax.set_facecolor(BG)
    img = Image.open(path)
    ax.imshow(np.array(img))
    ax.set_title(label, color=TEXT, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#30363d")

plt.tight_layout()
plt.savefig("outputs/gan_progression.png", dpi=150, facecolor=BG, bbox_inches="tight")
plt.close()
print("  Saved: outputs/gan_progression.png")

# === 4. VAE: original + reconstructed + morph in one image ===
print("4. VAE combined...")
fig = plt.figure(figsize=(16, 5), facecolor=BG)
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.15)

for idx, (path, title) in enumerate([
    ("outputs/vae_random_samples.png", "Random Samples"),
    ("outputs/vae_latent_space.png", "2D Latent Space"),
    ("outputs/vae_morph_0_to_9.png", "Morph: 0 → 9"),
]):
    ax = fig.add_subplot(gs[idx])
    ax.set_facecolor(BG)
    img = Image.open(path)
    ax.imshow(np.array(img))
    ax.set_title(title, color=TEXT, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#30363d")

fig.suptitle("VAE — Variational Autoencoder Results",
             fontsize=18, color=TEXT, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/vae_combined.png", dpi=150, facecolor=BG, bbox_inches="tight")
plt.close()
print("  Saved: outputs/vae_combined.png")

# === 5. Adversarial combined ===
print("5. Adversarial combined...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle("Adversarial Attacks on 99% Accurate CNN",
             fontsize=18, color=TEXT, fontweight="bold", y=1.02)

for ax, (path, title) in zip(axes, [
    ("outputs/adversarial_fgsm.png", "FGSM Attack\n(orig | adv | perturbation x5)"),
    ("outputs/targeted_attack_to_3.png", "Targeted → '3'\n(orig | fooled)"),
]):
    ax.set_facecolor(BG)
    img = Image.open(path)
    ax.imshow(np.array(img))
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#30363d")

plt.tight_layout()
plt.savefig("outputs/adversarial_combined.png", dpi=150, facecolor=BG, bbox_inches="tight")
plt.close()
print("  Saved: outputs/adversarial_combined.png")

# === 6. CNN visualization combined ===
print("6. CNN visualization combined...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
fig.suptitle("Inside the CNN — Filters, Activations & Dream Digits",
             fontsize=18, color=TEXT, fontweight="bold", y=1.02)

for ax, (path, title) in zip(axes, [
    ("outputs/conv1_filters.png", "Conv1 Filters (32 x 3x3)"),
    ("outputs/activations_conv1.png", "Activation Maps"),
    ("outputs/ideal_digits.png", "Dream Digits"),
]):
    ax.set_facecolor(BG)
    img = Image.open(path)
    ax.imshow(np.array(img))
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#30363d")

plt.tight_layout()
plt.savefig("outputs/cnn_internals.png", dpi=150, facecolor=BG, bbox_inches="tight")
plt.close()
print("  Saved: outputs/cnn_internals.png")

print("\nAll done!")
