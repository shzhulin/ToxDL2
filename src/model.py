import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from parameters.test_000 import device


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=True))
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=True)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), 0.5, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), 0.5, training=self.training)
        x = self.conv2(x, edge_index)
        pool = global_mean_pool(x, batch)
        return F.normalize(pool, dim=1)


class ToxDL_GCN_Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        protein_dim1 = 1280
        protein_dim2 = 512
        protein_dim3 = 256
        self.protein_GCN = GCN(protein_dim1, protein_dim2, protein_dim3)

        self.combine = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, data):
        # compute the protein embeddings using the protein embedder on the protein data of the batch
        protein_emb = self.protein_GCN(data.x, data.edge_index, data.batch)
        prot_domain = torch.Tensor(data.vector).to(device)
        # concatenate both embeddings and return the output of the FNN
        combined = torch.cat((protein_emb, prot_domain), dim=1)
        return self.combine(combined)
