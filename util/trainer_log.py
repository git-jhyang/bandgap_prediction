import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

MSELoss = nn.MSELoss()
MAELoss = nn.L1Loss()

def train(model, opt, data_loader, criterion):
    model.train()
    train_loss = 0
    train_ae   = 0
    n_data     = 0

    for batch in data_loader:
        n_batch = batch[8].shape[0]
        n_data += n_batch

#        pred, mtype = model(*batch[0:7])
#        loss = criterion(batch[7], pred) + criterion(batch[8], mtype)

        pred = model(*batch[0:8])
        loss = criterion(-torch.log(batch[8]/100 + 1e-4), pred)

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss.detach().item()
        train_ae   += MAELoss(batch[8], 100*torch.exp(-pred) - 0.01).detach().item()*n_batch

    return train_loss/len(data_loader), train_ae/n_data

def test(model, data_loader, criterion):
    model.eval()

    valid_loss = 0
    valid_ae   = 0
    valid_se   = 0
    n_data     = 0

    list_ids     = list()
    list_targets = list()
    list_preds   = list()

    with torch.no_grad():
        for batch in data_loader:
            n_batch = batch[8].shape[0]
            n_data += n_batch

            pred = model(*batch[0:8])

            if criterion is not None:
                loss = criterion(-torch.log(batch[8]/100 + 1e-4), pred).detach().item()
                valid_loss += loss
            valid_ae   += MAELoss(batch[8], 100*torch.exp(-pred) - 0.01).detach().item()*n_batch
            valid_se   += MSELoss(batch[8], 100*torch.exp(-pred) - 0.01).detach().item()*n_batch

            list_ids.append(batch[9].reshape(-1,1))
            list_targets.append(batch[8].cpu().numpy().reshape(-1,1))
            list_preds.append((100*torch.exp(-pred) - 0.01).cpu().numpy())
    
    loss = valid_loss/len(data_loader)
    mae  = valid_ae/n_data
    rmse = np.sqrt(valid_se/n_data)
    ids  = np.vstack(list_ids)
    targets = np.vstack(list_targets)
    preds   = np.vstack(list_preds)
    return loss, mae, rmse, ids, targets, preds   



def collate_fn(batch):
    list_atom_feat = list()
    list_rdf_feat  = list()
    list_bdf_feat  = list()
    list_atom_idx  = list()
    list_ele1_idx  = list()
    list_ele2_idx  = list()
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
        list_ele1_idx.append(data.idx_ele1 + base_ele_idx)
        list_ele2_idx.append(data.idx_ele2 + base_ele_idx)
        list_nbr_idx.append(data.idx_nbr + base_atom_idx)
        list_graph_idx.extend([i]*data.atom_feature.shape[0])

        list_gga.append(data.gap_gga)
        list_hse.append(data.gap_hse)

        list_ids.append(data.id)

        base_atom_idx += torch.max(data.idx_atom) + 1
        base_ele_idx += len(data.element)
    
    atom_feat = torch.cat(list_atom_feat, dim=0).float().cuda()
    rdf_feat  = torch.cat(list_rdf_feat, dim=0).float().cuda()
    bdf_feat  = torch.cat(list_bdf_feat, dim=0).float().cuda()
    atom_idx  = torch.cat(list_atom_idx, dim=0).view(-1).long().cuda()
    ele1_idx   = torch.cat(list_ele1_idx, dim=0).view(-1).long().cuda()
    ele2_idx   = torch.cat(list_ele2_idx, dim=0).view(-1).long().cuda()
    gap_gga   = torch.cat(list_gga, dim=0).view(-1,1).float().cuda()
    gap_hse   = torch.cat(list_hse, dim=0).view(-1,1).float().cuda()
    graph_idx = torch.tensor(list_graph_idx).view(-1).long().cuda()
    ids       = np.vstack(list_ids).reshape(-1,1)
    
    return atom_feat, rdf_feat, bdf_feat, atom_idx, ele1_idx, ele2_idx, graph_idx, \
           gap_gga, gap_hse, ids