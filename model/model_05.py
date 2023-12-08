import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class DistLayer(nn.Module):
    def __init__(self, n_atom_embed, n_dist_embed):
        super(DistLayer, self).__init__()

        self.n_ae = n_atom_embed
        self.n_de = n_dist_embed

        self.fc1 = nn.Linear(self.n_ae*4 + self.n_de, self.n_ae*2)
        self.bn1 = nn.BatchNorm1d(self.n_ae*2)

    def forward(self, x, atom_idx, ele_idx):
#        atom_feat = global_mean_pool(x[..., :self.n_ae], atom_idx)[atom_idx, :] # (N, n_atom_embed)
        x_atom = x[..., :self.n_ae]
        x_ele  = x[..., self.n_ae:self.n_ae*2] # (N, n_atom_embed)
        x_dist = x[..., self.n_ae*2:]
#        h = torch.cat([atom_feat, ele_feat, x[..., self.n_ae*2:]], dim=1) # (N, n_atom_embed*2 + n_bdf_embed + n_rdf_embed)

        h = torch.cat(
            [
                x_atom,
                global_mean_pool(x_atom, atom_idx)[atom_idx, :],
                x_ele,
                global_mean_pool(x_ele, atom_idx)[ele_idx, :],
                x_dist
            ], dim=1
        )

#        h = F.relu(self.fc2(h))
        h = F.relu(self.bn1(self.fc1(h)))
#        h = torch.cat([h1, h2], dim=1)
#        h = F.relu(torch.cat([h, h], dim=1) + x)
#        h = torch.cat([h, h], dim=1) 
        return h

class ModuleDistLayers(nn.Module):
    def __init__(self, n_atom_embed, n_rdf_embed, n_bdf_embed):
        super(ModuleDistLayers, self).__init__()
        self.n_ae = n_atom_embed
        self.n_re = n_rdf_embed
        self.n_be = n_bdf_embed

        self.rdf_layer = DistLayer(self.n_ae, self.n_re).cuda()
        self.bdf_layer = DistLayer(self.n_ae, self.n_be).cuda()

        self.fc1 = nn.Linear(self.n_ae*4, self.n_ae*2)
        self.bn1 = nn.BatchNorm1d(self.n_ae*2)

    def forward(self, x, rdf_feat, bdf_feat, atom_idx, ele_idx):
        x1 = self.rdf_layer(torch.cat([x, rdf_feat], dim=1), atom_idx, ele_idx)
        x2 = self.bdf_layer(torch.cat([x, bdf_feat], dim=1), atom_idx, ele_idx)

        h = F.relu(self.bn1(self.fc1(torch.cat([x1, x2], dim=1))))
#        x2 = F.softplus(x2)
#        h = F.relu(x1 + x2)
        return h

class DistNN(nn.Module):
    def __init__(self, n_atom_feat, n_rdf_feat, n_bdf_feat, dim_out):
        super(DistNN, self).__init__()
        self.n_af = n_atom_feat
        self.n_rf  = n_rdf_feat
        self.n_bf  = n_bdf_feat

        n_atom_embed = 32
        n_rdf_embed  = 128
        n_bdf_embed  = 128

        self.embed_atom = nn.Linear(self.n_af, n_atom_embed)
        self.embed_ele  = nn.Linear(self.n_af, n_atom_embed)
        self.embed_rdf  = nn.Linear(self.n_rf, n_rdf_embed)
        self.embed_bdf  = nn.Linear(self.n_bf, n_bdf_embed)

#        self.dist_layers = [ModuleDistLayers(n_atom_embed, n_rdf_embed, n_bdf_embed).cuda() for _ in range(n_module)]
        self.layer_1 = ModuleDistLayers(n_atom_embed, n_rdf_embed, n_bdf_embed).cuda()
        self.layer_2 = ModuleDistLayers(n_atom_embed, n_rdf_embed, n_bdf_embed).cuda()
        self.layer_3 = ModuleDistLayers(n_atom_embed, n_rdf_embed, n_bdf_embed).cuda()

#        self.fc1 = nn.Linear(n_atom_embed*2 + n_rdf_embed + n_bdf_embed, dim_out)
        self.fc1 = nn.Linear(n_atom_embed*2, dim_out)

        self.fc2 = nn.Linear(16, dim_out)
#        self.fc2 = nn.Linear(4, dim_out)

    def forward(self, atom_feat, rdf_feat, bdf_feat, atom_idx, ele_idx, graph_idx, ref_feat):
        x_atom = F.relu(self.embed_atom(atom_feat[..., :self.n_af]))
        x_ele  = F.relu(self.embed_atom(atom_feat[..., self.n_af:]))
        x_rdf  = F.relu(self.embed_rdf(rdf_feat))
        x_bdf  = F.relu(self.embed_bdf(bdf_feat))

        x = torch.cat([x_atom, x_ele], dim=1)

        h = self.layer_1(x, x_rdf, x_bdf, atom_idx, ele_idx)
        h = self.layer_2(h, x_rdf, x_bdf, atom_idx, ele_idx)
        h = self.layer_3(h, x_rdf, x_bdf, atom_idx, ele_idx)

#        h = torch.cat([h, x_rdf, x_bdf], dim=1)
        h = global_mean_pool(h, graph_idx)
        h = F.relu(self.fc1(h))
   #     h = F.relu(self.fc2(h))
        
        return h