import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

debug = False

class DistLayer(nn.Module):
    def __init__(self, n_dist_embed, n_atom_embed):
        super(DistLayer, self).__init__()

        self.n_de = n_dist_embed # distribution
        self.n_ae = n_atom_embed

        self.fc1 = nn.Linear(self.n_de + self.n_ae*2, self.n_ae*2)
        self.bn1 = nn.BatchNorm1d(self.n_ae*2)
        
    def forward(self, x, dist_feat, atom_idx, ele_idx):
        h = torch.cat([
                dist_feat,
                F.relu(global_mean_pool(x[..., :self.n_ae], atom_idx))[atom_idx, :],
                F.relu(global_mean_pool(x[..., self.n_ae:], ele_idx))[ele_idx, :]
            ], dim=1)
        h = self.fc1(h)
        h = self.bn1(h)
        out = F.relu(h + x)
        return out

class ModuleDistLayers(nn.Module):
    def __init__(self, n_atom_embed, n_rdf_embed, n_bdf_embed):
        super(ModuleDistLayers, self).__init__()
        self.n_ae = n_atom_embed
        self.n_re = n_rdf_embed
        self.n_be = n_bdf_embed

        self.rdf_layer = DistLayer(self.n_re, self.n_ae).cuda()
        self.bdf_layer = DistLayer(self.n_be, self.n_ae).cuda()

        self.fc1 = nn.Linear(self.n_ae*4, self.n_ae*2)

    def forward(self, x, rdf_feat, bdf_feat, atom_idx, ele_idx):
        x1 = self.rdf_layer(x, rdf_feat, atom_idx, ele_idx)
        x2 = self.bdf_layer(x, bdf_feat, atom_idx, ele_idx)

        out = F.relu(self.fc1(torch.cat([x1, x2], dim=1)))
        return out

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

        self.dist_layer_1 = ModuleDistLayers(n_atom_embed, n_rdf_embed, n_bdf_embed).cuda()
        self.dist_layer_2 = ModuleDistLayers(n_atom_embed, n_rdf_embed, n_bdf_embed).cuda()

        self.fc1 = nn.Linear(n_atom_embed*2 + 1, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, atom_feat, rdf_feat, bdf_feat, atom_idx, ele_idx, graph_idx, ref_feat):
        atom_1 = F.relu(self.embed_atom(atom_feat[..., :self.n_af]))
        atom_2 = F.relu(self.embed_atom(atom_feat[..., self.n_af:]))
        x_rdf  = F.relu6(self.embed_rdf(rdf_feat))
        x_bdf  = F.relu6(self.embed_bdf(bdf_feat))

        h = torch.cat([atom_1, atom_2], dim=1)

        h = self.dist_layer_1(h, x_rdf, x_bdf, atom_idx, ele_idx)
        h = self.dist_layer_2(h, x_rdf, x_bdf, atom_idx, ele_idx)

        h = global_mean_pool(h, graph_idx)
        h = F.relu(self.fc1(torch.cat([h, ref_feat], dim=1)))
        gap = self.fc2(h)
        return gap
