import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class DistNN(nn.Module):
    def __init__(self, n_atom_feat, n_rdf_feat, n_bdf_feat):
        super(DistNN, self).__init__()
        self.n_af = n_atom_feat
        self.n_rf = n_rdf_feat        
        self.n_bf = n_bdf_feat

        n_atom_embed = 128
        n_rdf_embed  = 128
        n_bdf_embed  = 128

        self.embed_atom  = nn.Linear(self.n_af, n_atom_embed)
        self.embed_rdf   = nn.Linear(self.n_rf, n_rdf_embed)
        self.embed_bdf   = nn.Linear(self.n_bf, n_bdf_embed)

        self.fc_rdf = nn.Linear(n_atom_embed*2 + n_rdf_embed, n_atom_embed)
        self.fc_bdf = nn.Linear(n_atom_embed*2 + n_rdf_embed, n_atom_embed)

        self.bn1 = nn.BatchNorm1d(n_atom_embed*2)

        self.fc1 = nn.Linear(n_atom_embed*2 + 1, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, atom_feat, rdf_feat, bdf_feat, atom_idx, ele1_idx, ele2_idx, graph_idx, ref_feat):
        atom_1 = F.relu(self.embed_atom(atom_feat[..., :self.n_af]))
        atom_2 = F.relu(self.embed_atom(atom_feat[..., self.n_af:]))
        x_rdf  = F.relu6(self.embed_rdf(rdf_feat))
        x_bdf  = F.relu6(self.embed_bdf(bdf_feat))

        h1 = torch.cat([atom_1, atom_2, x_rdf], dim=1)
        h2 = torch.cat([atom_1, atom_2, x_bdf], dim=1)

        h = F.relu(self.bn1(torch.cat([self.fc_rdf(h1), self.fc_bdf(h2)], dim=1)))
        
        h = global_mean_pool(h, graph_idx)
        h = F.relu(self.fc1(torch.cat([h, ref_feat], dim=1)))
        gap = F.relu(self.fc2(h))
        return gap
#        log = F.relu(self.fc3(h))
#        return gap, log
