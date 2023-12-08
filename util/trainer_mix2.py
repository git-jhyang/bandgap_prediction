import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

MSELoss = nn.MSELoss()

def train(model, opt, data_loader, criterion):
    model.train()
    train_loss = 0
    train_mae  = 0
    n_data     = len(data_loader)
    for batch in data_loader:
#        pred, mtype = model(*batch[0:7])
#        loss = criterion(batch[7], pred) + criterion(batch[8], mtype)
        pred = model(*batch[0:7])

        mae = criterion(batch[7], pred)
        log_mae = criterion(-torch.log(batch[7]/100 + 1e-4*torch.rand(batch[7].shape[0]).cuda()), 
                            -torch.log(pred/100 + 1e-4*torch.rand(pred.shape[0]).cuda()))

        loss = mae + log_mae

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            train_loss += loss.detach().item()
            train_mae  += mae.detach().item()
    return train_loss/n_data, train_mae/n_data

def test(model, data_loader, criterion):
    model.eval()

    valid_loss = 0
    valid_mae  = 0

    list_ids     = list()
    list_targets = list()
    list_preds   = list()

    n_data = len(data_loader)

    with torch.no_grad():
        for batch in data_loader:
            pred = model(*batch[0:7])

            mae  = criterion(batch[7], pred).detach().item()
            log_mae = criterion(-torch.log(batch[7]/100 + 1e-4), 
                                -torch.log(pred/100 + 1e-4)).detach().item()

            loss = mae + log_mae

            valid_loss += loss
            valid_mae  += mae

            list_ids.append(batch[8].cpu().numpy().reshape(-1,1))
            list_targets.append(batch[7].cpu().numpy().reshape(-1,1))
            list_preds.append(pred.cpu().numpy().reshape(-1,1))
    
    return valid_loss/n_data, valid_mae/n_data, \
           np.vstack(list_ids), np.vstack(list_targets), np.vstack(list_preds)


def collate_fn(batch):
    list_atom_feat = list()
    list_rdf_feat  = list()
    list_bdf_feat  = list()
    list_atom_idx  = list()
    list_ele_idx   = list()
    list_nbr_idx   = list()
    list_graph_idx = list()
    list_gga       = list()
    list_hse       = list()
    list_ids       = list()

    base_atom_idx  = 0
    base_ele_idx   = 0

    for i, data in enumerate(batch):
        list_atom_feat.append(data.atom_feature)
        list_rdf_feat.append(data.rdf_feature)
        list_bdf_feat.append(data.bdf_feature)

        list_atom_idx.append(data.idx_atom + base_atom_idx)
        list_ele_idx.append(data.idx_ele2 + base_ele_idx)
        list_nbr_idx.append(data.idx_nbr + base_atom_idx)
        list_graph_idx.extend([i]*data.atom_feature.shape[0])

        list_gga.append(data.gap_gga)
        list_hse.append(data.gap_hse)

        list_ids.append(data.id)

        base_atom_idx += data.idx_atom[-1] + 1
        base_ele_idx += len(data.element)
    
    atom_feat = torch.cat(list_atom_feat, dim=0).float().cuda()
    rdf_feat  = torch.cat(list_rdf_feat, dim=0).float().cuda()
    bdf_feat  = torch.cat(list_bdf_feat, dim=0).float().cuda()
    atom_idx  = torch.cat(list_atom_idx, dim=0).view(-1).long().cuda()
    ele_idx   = torch.cat(list_ele_idx, dim=0).view(-1).long().cuda()
    gap_gga   = torch.cat(list_gga, dim=0).view(-1,1).float().cuda()
    gap_hse   = torch.cat(list_hse, dim=0).view(-1,1).float().cuda()
    graph_idx = torch.tensor(list_graph_idx).view(-1).long().cuda()
    ids       = torch.tensor(list_ids).view(-1,1).cuda()
    
    return atom_feat, rdf_feat, bdf_feat, atom_idx, ele_idx, graph_idx, \
           gap_gga, gap_hse, ids