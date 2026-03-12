"""Generate GNN graph visualization and RL progression image"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import os

# === GNN: Show a sample MNIST digit as graph ===
from torchvision import datasets, transforms
import torch

print("Generating GNN graph visualization...")
transform = transforms.ToTensor()
mnist = datasets.MNIST("data", train=False, download=True, transform=transform)

fig, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor="black")
fig.suptitle("GNN-GAT: MNIST Pixels as Graph Nodes (98.36% accuracy)",
             fontsize=20, color="white", fontweight="bold")

for idx in range(10):
    # Find a digit of class idx
    for img, label in mnist:
        if label == idx:
            break

    ax = axes[idx // 5, idx % 5]
    ax.set_facecolor("black")

    img_np = img.squeeze().numpy()
    threshold = 0.15

    # Draw edges first (faint)
    coords = []
    for i in range(28):
        for j in range(28):
            if img_np[i, j] > threshold:
                coords.append((j, 27 - i, img_np[i, j]))
                # Draw edges to neighbors
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 28 and 0 <= nj < 28 and img_np[ni, nj] > threshold:
                        ax.plot([j, nj], [27-i, 27-ni], color="#1a3a5c", linewidth=0.3, alpha=0.4)

    # Draw nodes
    if coords:
        xs, ys, vals = zip(*coords)
        ax.scatter(xs, ys, c=vals, cmap="hot", s=8, vmin=0, vmax=1, zorder=2)

    n_nodes = len(coords)
    ax.set_title(f"Digit {idx} ({n_nodes} nodes)", color="white", fontsize=14)
    ax.set_xlim(-1, 28)
    ax.set_ylim(-1, 28)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333333")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("outputs/gnn_graph_viz.png", dpi=150, facecolor="black", bbox_inches="tight")
print("Saved: outputs/gnn_graph_viz.png")
plt.close()

# === RL: Show progression over training ===
print("Generating RL progression...")
rl_files = sorted([f for f in os.listdir("outputs") if f.startswith("rl_writer_ep0") and f.endswith(".png")])
# Pick key milestones
milestones = ["rl_writer_ep00500.png", "rl_writer_ep01000.png", "rl_writer_ep02000.png",
              "rl_writer_ep03000.png", "rl_writer_ep04000.png", "rl_writer_ep05000.png"]
milestones = [f for f in milestones if f in os.listdir("outputs")]

if milestones:
    fig, axes = plt.subplots(1, len(milestones), figsize=(4 * len(milestones), 4), facecolor="black")
    fig.suptitle("RL Digit Writer: Learning to Draw '7' (PPO, 5000 episodes)",
                 fontsize=18, color="white", fontweight="bold")

    for i, fname in enumerate(milestones):
        ax = axes[i] if len(milestones) > 1 else axes
        img = Image.open(f"outputs/{fname}")
        ax.imshow(np.array(img), cmap="gray")
        ep = fname.replace("rl_writer_ep", "").replace(".png", "")
        ax.set_title(f"Episode {int(ep)}", color="white", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_color("#333333")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig("outputs/rl_progression.png", dpi=150, facecolor="black", bbox_inches="tight")
    print("Saved: outputs/rl_progression.png")
    plt.close()

print("Done!")
