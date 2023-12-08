import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

MSELoss = nn.MSELoss()

def train(model, opt, data_loader, criterion):
    model.train()
    train_loss = 0
    train_mae  = 0
    train_f1   = 0
    n_data     = len(data_loader)

    for batch in data_loader:
        pred, mtype = model(*batch[0:7])
        loss1 = criterion(batch[7], pred) 
        loss2 = criterion(batch[8], mtype)

        opt.zero_grad()
        (loss1 + loss2).backward()
        opt.step()

        train_mae  += loss1.detach().item()
        train_f1   += f1_score(batch[8].cpu().int(), (mtype > 0.5).cpu().int(), average='weighted')
        train_loss += (loss1 + loss2).detach().item()

    return train_loss/n_data, train_mae/n_data, train_f1/n_data

def test(model, data_loader, criterion):
    model.eval()

    valid_loss = 0
    valid_mae  = 0
    valid_f1   = 0

    list_ids     = list()
    list_targets = list()
    list_preds   = list()
    n_data       = len(data_loader)

    with torch.no_grad():
        for batch in data_loader:
            pred, mtype = model(*batch[0:7])
            loss1 = criterion(batch[7], pred).detach().item() 
            loss2 = criterion(2*batch[8], 2*mtype).detach().item()

            valid_loss += loss1 + loss2
            valid_mae  += loss1
            valid_f1   += f1_score(batch[8].cpu().int(), (mtype > 0.5).cpu().int(), average='weighted')

            list_ids.append(batch[9].cpu().numpy().reshape(-1,1))
            list_targets.append(batch[7].cpu().numpy().reshape(-1,1))
            list_preds.append(pred.cpu().numpy())
    
    return valid_loss/n_data, valid_mae/n_data, valid_f1/n_data, \
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
    list_mtype     = list()
    base_atom_idx  = 0
    base_ele_idx   = 0

    for i, data in enumerate(batch):
        list_atom_feat.append(data.atom_feature)
        list_rdf_feat.append(data.rdf_feature)
        list_bdf_feat.append(data.bdf_feature)

        list_atom_idx.append(data.idx_atom + base_atom_idx)
        list_ele_idx.append(data.idx_ele + base_ele_idx)
        list_nbr_idx.append(data.idx_nbr + base_atom_idx)
        list_graph_idx.extend([i]*data.atom_feature.shape[0])

        list_gga.append(data.gap_gga)
        list_hse.append(data.gap_hse)
        mtype = 1 if data.gap_hse > 0 else 0
        list_mtype.append(mtype)
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
    mat_type  = torch.tensor(list_mtype, dtype=torch.float).view(-1,1).cuda()
    graph_idx = torch.tensor(list_graph_idx).view(-1).long().cuda()
    ids       = torch.tensor(list_ids).view(-1,1).cuda()
    
    return atom_feat, rdf_feat, bdf_feat, atom_idx, ele_idx, graph_idx, \
           gap_gga, gap_hse, mat_type, ids