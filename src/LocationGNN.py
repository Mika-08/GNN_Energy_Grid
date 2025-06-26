import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import random

from GNN import PowerFlowGNN

def generate_instance():
    num_locations = random.randint(3, 5)
    locations = [f"Loc{i}" for i in range(num_locations)]

    node_features = []
    node_targets = []
    edge_index = []
    edge_attr = []

    loc_indices = {}
    supply_indices = {}
    demand_indices = {}
    node_types = []

    for loc in locations:
        loc_idx = len(node_features)
        loc_indices[loc] = loc_idx
        node_features.append([0.0, 0.0])
        node_targets.append(0.0)
        node_types.append("Location")

    for loc in locations:
        demand = random.uniform(3, 8)
        supply = random.uniform(2, 7)

        if supply > 0:
            supply_idx = len(node_features)
            supply_indices[loc] = supply_idx
            node_features.append([0.0, supply])
            node_targets.append(supply)
            node_types.append("Supply")
            edge_index.append([supply_idx, loc_indices[loc]])
            edge_attr.append([random.uniform(0.5, 2.0)])

        if demand > 0:
            demand_idx = len(node_features)
            demand_indices[loc] = demand_idx
            node_features.append([demand, 0.0])
            node_targets.append(0.0)
            node_types.append("Demand")
            edge_index.append([loc_indices[loc], demand_idx])
            edge_attr.append([random.uniform(0.5, 2.0)])

        node_features[loc_indices[loc]] = [0.0, 0.0]

    for i in range(num_locations):
        for j in range(num_locations):
            if i != j and random.random() < 0.7:
                edge_index.append([loc_indices[locations[i]], loc_indices[locations[j]]])
                edge_attr.append([random.uniform(0.5, 2.0)])

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_targets, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_y = edge_attr.squeeze()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_y=edge_y, node_types=node_types)

def train(model, optimizer, n_epochs=500, batch_size=32):
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Generate batch of graphs
        data_list = [generate_instance() for _ in range(batch_size)]
        batch = Batch.from_data_list(data_list)

        pred_node, pred_edge = model(batch.x, batch.edge_index, batch.edge_attr)

        # Node loss: production prediction vs target
        loss_node = F.mse_loss(pred_node, batch.y)

        # Create masks for edges with zero and non-zero true flow
        zero_flow_mask = (batch.edge_y == 0)
        nonzero_flow_mask = (batch.edge_y != 0)

        # Edge loss: only calculate where flow expected
        if nonzero_flow_mask.sum() > 0:
            loss_edge_flow = F.mse_loss(pred_edge[nonzero_flow_mask], batch.edge_y[nonzero_flow_mask])
        else:
            loss_edge_flow = 0.0

        # Strong penalty on edges where flow should be zero
        if zero_flow_mask.sum() > 0:
            loss_edge_zero = F.mse_loss(pred_edge[zero_flow_mask], batch.edge_y[zero_flow_mask])
        else:
            loss_edge_zero = 0.0

        loss_edge = loss_edge_flow + 10 * loss_edge_zero  # weighted zero flow penalty

        loss = loss_node + loss_edge

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:03d} | Node Loss: {loss_node.item():.4f} | Edge Loss: {loss_edge.item():.4f}")


def main():
    model = PowerFlowGNN(in_channels=2, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train(model, optimizer)

    model.eval()
    test_data = generate_instance()
    with torch.no_grad():
        pred_node, pred_edge = model(test_data.x, test_data.edge_index, test_data.edge_attr)

    print("\n=== Test Data ===")
    for i, (pred, target) in enumerate(zip(pred_node, test_data.y)):
        print(f"  Node {i} - Predicted: {pred.item():.2f}, Target: {target.item():.2f}")

    print("\n=== Edge Flows ===")
    for i in range(test_data.edge_index.shape[1]):
        src = test_data.edge_index[0, i].item()
        dst = test_data.edge_index[1, i].item()
        pred = pred_edge[i].item()
        actual = test_data.edge_y[i].item()
        print(f"  {src} → {dst}: Predicted: {pred:.2f}, Target: {actual:.2f}")

if __name__ == "__main__":
    main()