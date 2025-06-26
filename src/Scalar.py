from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import numpy as np
import torch


class GraphScalar:
    def __init__(self):
        '''
        Scaler options thought of so far:
        - MinMaxScaler
        - StandardScaler(with_mean= False)
        - MaxAbsScaler()
        - RobustScaler()
        '''
        self.NodeFeatScalar = MaxAbsScaler()
        #self.NodeFeatScalar = MinMaxScaler(feature_range=(0, 1))
        self.EdgeFeatScalar = MaxAbsScaler()
        self.NodeGtScalar = MaxAbsScaler()
        # self.NodeGtScalar = MinMaxScaler(feature_range=(0, 1))
        self.EdgeGtScalar = MaxAbsScaler()
        
    
    def normalize_node_features(self, node_features):
        """
        Scale node features using StandardScaler.

        Do not scale the availability feature. Which is the final feature

        """
        T, N, F = node_features.shape

        flat_node = node_features[:, :, :3].reshape(-1, 3)  # [T*N, 3]
        print("flat_node shape", flat_node.shape)
        self.NodeFeatScalar.fit(flat_node)
        scaled_node_part = self.NodeFeatScalar.transform(flat_node).reshape(T, N, 3)
        capacity_column = node_features[:, :, 3:]  # shape: [T, N, 1]

        # Concatenate back
        node_feature_scaled = np.concatenate([scaled_node_part, capacity_column], axis=2)
        return torch.tensor(node_feature_scaled, dtype=torch.float)
         
    
    def normalize_edge_features(self, edge_features):
        T, E, F = edge_features.shape
        flat_edge = edge_features.reshape(-1, 1)
        self.EdgeFeatScalar.fit(flat_edge)
        edge_feature_scaled = self.EdgeFeatScalar.transform(flat_edge).reshape(T, E, 1)
        return torch.tensor(edge_feature_scaled, dtype=torch.float)

    def normalize_node_gt(self, node_gt):
        T, N, _ = node_gt.shape
        flat_node_gt = node_gt.reshape(-1, 1)  # Fix the shape
        self.NodeGtScalar.fit(flat_node_gt)
        node_gt_scaled = self.NodeGtScalar.transform(flat_node_gt).reshape(T, N, 1)
        return torch.tensor(node_gt_scaled, dtype=torch.float)

    def normalize_edge_gt(self, edge_gt):
        T, E = edge_gt.shape
        flat_edge_gt = edge_gt.reshape(-1, 1)
        self.EdgeGtScalar.fit(flat_edge_gt)
        edge_gt_scaled = self.EdgeGtScalar.transform(flat_edge_gt).reshape(T, E)
        return torch.tensor(edge_gt_scaled, dtype=torch.float)
    
    def inverse_node_prediction(self, node_prediction):
        # Inverse node prediction
        return self.NodeGtScalar.inverse_transform(node_prediction.unsqueeze(-1).cpu().numpy())

    def inverse_edge_prediction(self, edge_prediction):
        # Inverse edge prediction
        return self.EdgeGtScalar.inverse_transform(edge_prediction.unsqueeze(-1).cpu().numpy())

    def normalize(self, node_features, edge_features, node_gt, edge_gt):
        # Normalize node features
        node_features = self.normalize_node_features(node_features)
        # Normalize edge features
        edge_features = self.normalize_edge_features(edge_features)
        # Normalize node gt
        node_gt = self.normalize_node_gt(node_gt)
        # Normalize edge gt
        edge_gt = self.normalize_edge_gt(edge_gt)

        return node_features, edge_features, node_gt, edge_gt
    
    def inverse(self, node_prediction, edge_prediction):
        # Inverse node prediction
        node_prediction = self.inverse_node_prediction(node_prediction)
        # Inverse edge prediction
        edge_prediction = self.inverse_edge_prediction(edge_prediction)

        return node_prediction, edge_prediction