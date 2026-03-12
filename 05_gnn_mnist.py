"""
Step 5: GNN for MNIST — ピクセルをグラフのノードとして分類
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torchvision import datasets, transforms
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=== Step 5: GNN MNIST ===")
print("Converting MNIST pixels to graphs...")


def mnist_to_graph(image, label, threshold=0.2):
    """
    28x28 image -> graph
    - Node: each pixel with intensity > threshold
    - Edge: 8-neighbor connectivity
    - Node feature: [intensity, x_norm, y_norm]
    """
    img = image.squeeze().numpy()
    coords = []
    features = []

    for i in range(28):
        for j in range(28):
            if img[i, j] > threshold:
                coords.append((i, j))
                features.append([img[i, j], i / 27.0, j / 27.0])

    if len(coords) == 0:
        # Empty image fallback
        coords = [(14, 14)]
        features = [[0.0, 0.5, 0.5]]

    n = len(coords)
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


# Convert dataset (subset for speed)
transform = transforms.ToTensor()
train_mnist = datasets.MNIST("data", train=True, download=True, transform=transform)
test_mnist = datasets.MNIST("data", train=False, transform=transform)

N_TRAIN = 5000
N_TEST = 1000

print(f"Converting {N_TRAIN} train + {N_TEST} test images to graphs...")
t0 = time.time()
train_graphs = [mnist_to_graph(train_mnist[i][0], train_mnist[i][1]) for i in range(N_TRAIN)]
test_graphs = [mnist_to_graph(test_mnist[i][0], test_mnist[i][1]) for i in range(N_TEST)]
print(f"Conversion done in {time.time()-t0:.1f}s")
print(f"Example graph: {train_graphs[0].num_nodes} nodes, {train_graphs[0].num_edges} edges")

train_loader = PyGDataLoader(train_graphs, batch_size=64, shuffle=True)
test_loader = PyGDataLoader(test_graphs, batch_size=256)


class GNN_MNIST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.fc2(x)


model = GNN_MNIST().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\nTraining GNN...")
for epoch in range(1, 31):
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

    # Test
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
    print(f"Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc*100:.2f}%")

torch.save(model.state_dict(), "gnn_mnist.pth")
print("Saved: gnn_mnist.pth")
