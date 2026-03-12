"""
Generate a polished showcase image for README.
Clean 4x2 grid, consistent sizing, proper padding.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

# Load and resize all images to consistent dimensions
def load_and_pad(path, target_ratio=1.0):
    """Load image and return as numpy array."""
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except:
        return np.zeros((100, 100, 3), dtype=np.uint8)


fig = plt.figure(figsize=(20, 24), facecolor="#0d1117")

# Title area
fig.text(0.5, 0.97, "MNIST Deep Learning Playground",
         fontsize=32, color="white", fontweight="bold", ha="center",
         fontfamily="monospace")
fig.text(0.5, 0.955,
         "CNN 99.35%  |  VAE  |  DCGAN  |  DDPM  |  GNN-GAT 98.36%  |  Adversarial  |  t-SNE  |  RL-PPO",
         fontsize=12, color="#58a6ff", ha="center", fontfamily="monospace")

gs = gridspec.GridSpec(4, 2, hspace=0.28, wspace=0.12,
                       left=0.04, right=0.96, top=0.94, bottom=0.02)

panels = [
    (0, 0, "outputs/gan_final.png",
     "DCGAN — 200 epochs",
     "Hinge loss + spectral norm"),

    (0, 1, "outputs/diffusion_epoch030.png",
     "Diffusion Model (DDPM)",
     "Iterative denoising from Gaussian noise"),

    (1, 0, "outputs/vae_latent_space.png",
     "VAE — 2D Latent Space",
     "Smooth interpolation between digits"),

    (1, 1, "outputs/gnn_graph_viz.png",
     "GNN-GAT — 98.36% Accuracy",
     "Pixels as graph nodes with attention"),

    (2, 0, "outputs/adversarial_fgsm.png",
     "Adversarial Attack (FGSM)",
     "Original | Adversarial | Perturbation x5"),

    (2, 1, "outputs/targeted_attack_to_3.png",
     "Targeted Attack → '3'",
     "All digits misclassified as 3"),

    (3, 0, "outputs/tsne_features.png",
     "t-SNE Feature Embedding",
     "10,000 digits in 128-dim → 2D"),

    (3, 1, "outputs/rl_progression.png",
     "RL Digit Writer (PPO)",
     "Learning to draw '7' over 5000 episodes"),
]

for row, col, img_path, title, desc in panels:
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor("#0d1117")

    img = load_and_pad(img_path)
    ax.imshow(img)

    ax.set_title(title, fontsize=15, color="white", fontweight="bold",
                 pad=8, fontfamily="monospace")
    ax.text(0.5, -0.06, desc, transform=ax.transAxes,
            fontsize=11, color="#8b949e", ha="center", fontfamily="monospace")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#30363d")
        spine.set_linewidth(1)

plt.savefig("showcase.png", dpi=150, facecolor="#0d1117", edgecolor="none",
            bbox_inches="tight", pad_inches=0.4)
print("Saved: showcase.png")

plt.savefig("showcase_web.png", dpi=90, facecolor="#0d1117", edgecolor="none",
            bbox_inches="tight", pad_inches=0.4)
print("Saved: showcase_web.png")
