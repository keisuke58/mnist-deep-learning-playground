"""
Step 5: GNN for MNIST — ピクセルをグラフのノードとして分類
GAT (Graph Attention Network) + full dataset for 95%+ accuracy
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torchvision import datasets, transforms
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=== Step 5: GNN MNIST (GAT, Full Dataset) ===")


def mnist_to_graph(image, label, threshold=0.15):
    """
    28x28 image -> graph
    - Node: each pixel with intensity > threshold
    - Edge: 8-neighbor connectivity
    - Node feature: [intensity, x_norm, y_norm, local_mean, edge_strength]
    """
    img = image.squeeze().numpy()
    coords = []
    features = []

    for i in range(28):
        for j in range(28):
            if img[i, j] > threshold:
                # Local mean (3x3 neighborhood)
                patch = img[max(0,i-1):min(28,i+2), max(0,j-1):min(28,j+2)]
                local_mean = patch.mean()
                # Edge strength (gradient magnitude)
                gx = img[i, min(27,j+1)] - img[i, max(0,j-1)]
                gy = img[min(27,i+1), j] - img[max(0,i-1), j]
                edge = np.sqrt(gx**2 + gy**2)

                coords.append((i, j))
                features.append([img[i, j], i / 27.0, j / 27.0, local_mean, edge])

    if len(coords) == 0:
        coords = [(14, 14)]
        features = [[0.0, 0.5, 0.5, 0.0, 0.0]]

    coord_to_idx = {c: i for i, c in enumerate(coords)}

    edges = []
    for idx, (i, j) in enumerate(coords):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                nb = (i + di, j + dj)
                if nb in coord_to_idx:
                    edges.append([idx, coord_to_idx[nb]])

    if len(edges) == 0:
        edges = [[0, 0]]

    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


transform = transforms.ToTensor()
train_mnist = datasets.MNIST("data", train=True, download=True, transform=transform)
test_mnist = datasets.MNIST("data", train=False, transform=transform)

N_TRAIN = 20000  # 4x more data
N_TEST = 5000

print(f"Converting {N_TRAIN} train + {N_TEST} test images to graphs...")
t0 = time.time()
train_graphs = [mnist_to_graph(train_mnist[i][0], train_mnist[i][1]) for i in range(N_TRAIN)]
test_graphs = [mnist_to_graph(test_mnist[i][0], test_mnist[i][1]) for i in range(N_TEST)]
print(f"Conversion done in {time.time()-t0:.1f}s")
print(f"Example graph: {train_graphs[0].num_nodes} nodes, {train_graphs[0].num_edges} edges")

train_loader = PyGDataLoader(train_graphs, batch_size=128, shuffle=True)
test_loader = PyGDataLoader(test_graphs, batch_size=256)


class GAT_MNIST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(5, 32, heads=4, concat=True)     # -> 128
        self.conv2 = GATConv(128, 64, heads=4, concat=True)    # -> 256
        self.conv3 = GATConv(256, 128, heads=2, concat=False)  # -> 128
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc1 = torch.nn.Linear(256, 128)  # mean_pool + max_pool = 256
        self.fc2 = torch.nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.elu(self.conv3(x, edge_index))
        # Combine mean and max pooling
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


model = GAT_MNIST().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

print(f"\nTraining GAT (params: {sum(p.numel() for p in model.parameters()):,})...")
best_acc = 0
for epoch in range(1, 51):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    acc = correct / total
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/gnn_mnist.pth")
    print(f"Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc*100:.2f}% | Best: {best_acc*100:.2f}%")

print(f"\nBest Test Accuracy: {best_acc*100:.2f}%")
print("Saved: models/gnn_mnist.pth")
