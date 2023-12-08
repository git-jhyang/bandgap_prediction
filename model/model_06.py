import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

debug = False

class DistLayer(nn.Module):
    def __init__(self, n_atom_embed, n_dist_embed):
        super(DistLayer, self).__init__()

        self.n_ae = n_atom_embed
        self.n_de = n_dist_embed

        self.fc1 = nn.Linear(self.n_ae*2 + self.n_de, self.n_ae)
        self.bn1 = nn.BatchNorm1d(self.n_ae)

    def forward(self, x, atom_idx, ele_idx):
        h = F.relu(self.bn1(self.fc1(x)))
        h = torch.cat([h, global_mean_pool(h, ele_idx)[ele_idx, :]], dim=1)

        return h

class ModuleDistLayers(nn.Module):
    def __init__(self, n_atom_embed, n_dist_embed):
        super(ModuleDistLayers, self).__init__()
        self.n_ae = n_atom_embed
        self.n_de = n_dist_embed

        self.layer_1 = DistLayer(self.n_ae, self.n_de).cuda()
        self.layer_2 = DistLayer(self.n_ae, self.n_de).cuda()

        self.fc1 = nn.Linear(self.n_ae*2, self.n_ae)
        self.bn1 = nn.BatchNorm1d(self.n_ae)

    def forward(self, x, dist_feat, atom_idx, ele_idx):
        h = self.layer_1(torch.cat([x, dist_feat], dim=1), atom_idx, ele_idx)
        h = self.layer_2(torch.cat([h, dist_feat], dim=1), atom_idx, ele_idx)
        #out = self.fc1(torch.cat([x1, x2], dim=1))
#        x1 = F.relu(x1)
#        x2 = F.softplus(x2)
        out = F.relu(self.bn1(self.fc1(h)))
        return out

class DistNN(nn.Module):
    def __init__(self, n_atom_feat, n_rdf_feat, n_bdf_feat, dim_out):
        super(DistNN, self).__init__()
        self.n_atom_feat = n_atom_feat
        self.n_rdf_feat  = n_rdf_feat
        self.n_bdf_feat  = n_bdf_feat

        n_atom_embed = 128
        n_rdf_embed  = 256
        n_bdf_embed  = 256

        self.embed_atom = nn.Linear(self.n_atom_feat, n_atom_embed)
        self.embed_rdf  = nn.Linear(self.n_rdf_feat, n_rdf_embed)
        self.embed_bdf  = nn.Linear(self.n_bdf_feat, n_bdf_embed)

        self.rdf_layer = ModuleDistLayers(n_atom_embed, n_rdf_embed).cuda()
        self.bdf_layer = ModuleDistLayers(n_atom_embed, n_bdf_embed).cuda()

        self.fc1 = nn.Linear(n_atom_embed*2, dim_out)

    def forward(self, atom_feat, rdf_feat, bdf_feat, atom_idx, ele_idx, graph_idx, ref_feat):
        x_atom = F.relu(self.embed_atom(atom_feat[..., :self.n_atom_feat]))
        x_ele  = F.relu(self.embed_atom(atom_feat[..., self.n_atom_feat:]))
        x_rdf  = F.relu(self.embed_rdf(rdf_feat))
        x_bdf  = F.relu(self.embed_bdf(bdf_feat))

        x = torch.cat([x_atom, x_ele], dim=1)

        h1 = self.rdf_layer(x, x_rdf, atom_idx, ele_idx)
        h2 = self.bdf_layer(x, x_bdf, atom_idx, ele_idx)

        h = torch.cat([h1, h2], dim=1)
        h = global_mean_pool(h, graph_idx)
        h = self.fc1(h)
        return h