"""
Generate a single high-quality showcase image combining all MNIST experiments.
Layout: 4x2 grid with labels
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

fig = plt.figure(figsize=(24, 28), facecolor="black")
fig.suptitle("MNIST Deep Learning Playground", fontsize=36, color="white",
             fontweight="bold", y=0.98)
fig.text(0.5, 0.965,
         "8 experiments exploring CNN, VAE, GAN, Diffusion, GNN, Adversarial Attacks, Feature Visualization & RL",
         ha="center", fontsize=16, color="#aaaaaa")

gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.15,
                       left=0.03, right=0.97, top=0.94, bottom=0.02)

panels = [
    # (row, col, image_path, title, description)
    (0, 0, "outputs/gan_final.png",
     "1. DCGAN — Generative Adversarial Network",
     "200 epochs with hinge loss + spectral norm — crisp handwritten digits"),

    (0, 1, "outputs/vae_latent_space.png",
     "2. VAE — Variational Autoencoder",
     "2D latent space: smooth interpolation between digit classes"),

    (1, 0, "outputs/diffusion_epoch030.png",
     "3. Diffusion Model (DDPM)",
     "Iterative denoising generates digits from pure Gaussian noise"),

    (1, 1, "outputs/vae_morph_0_to_9.png",
     "4. VAE Morphing: 0 → 9",
     "Latent space interpolation smoothly transforms between digits"),

    (2, 0, "outputs/adversarial_fgsm.png",
     "5. Adversarial Attack (FGSM)",
     "Original | Adversarial | Perturbation ×5 — invisible noise fools 99% CNN"),

    (2, 1, "outputs/targeted_attack_to_3.png",
     "6. Targeted Attack → 3",
     "Original | Attacked — all digits misclassified as '3'"),

    (3, 0, "outputs/tsne_features.png",
     "7. t-SNE Feature Embedding",
     "10,000 test digits projected from 128-dim CNN features to 2D"),

    (3, 1, "outputs/ideal_digits.png",
     "8. Dream Digits — What CNN 'sees'",
     "Gradient ascent on class scores reveals the network's ideal digit"),
]

for row, col, img_path, title, desc in panels:
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor("black")

    try:
        img = mpimg.imread(img_path)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    except Exception as e:
        ax.text(0.5, 0.5, f"Missing:\n{img_path}", ha="center", va="center",
                color="red", fontsize=14, transform=ax.transAxes)

    ax.set_title(title, fontsize=18, color="#00ccff", fontweight="bold", pad=10)
    ax.set_xlabel(desc, fontsize=12, color="#888888", labelpad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.5)

# Footer
fig.text(0.5, 0.005,
         "CNN 99.35% | VAE 2D Latent | DCGAN 200ep | DDPM 30ep | GNN-GAT 98.36% | FGSM/PGD Attacks | t-SNE | DeepDream | RL-PPO Writer",
         ha="center", fontsize=13, color="#666666")

plt.savefig("showcase.png", dpi=150, facecolor="black", edgecolor="none",
            bbox_inches="tight", pad_inches=0.3)
print("Saved: showcase.png")

# Also make a smaller version for README
plt.savefig("showcase_web.png", dpi=80, facecolor="black", edgecolor="none",
            bbox_inches="tight", pad_inches=0.3)
print("Saved: showcase_web.png")
