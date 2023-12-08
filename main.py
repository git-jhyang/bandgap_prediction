import numpy as np
import pandas as pd
import os, gc, json
import torch.nn
from torch.utils.data import DataLoader
from util.AdaBound import AdaBound
from torch.utils.tensorboard import SummaryWriter
import time, shutil
from natsort import index_natsorted

def exec_model_until_success(
        model_type,
        dataset,
        mobj,
        tr,
        model_name,
        lr = 1e-5,
        wd = 1e-7,
        tries = 1,
        root_model = 'c:/WORKSPACE_KRICT/MODELS/202204/nmre',
        num_epochs = 300,
        batch_size = 128,
        train_ratio = 0.7,
        valid_ratio = 0.2,
        metal_ratio = 1,
        kmax = 5
        ):
    k = 0
    while True:
        success = exec_model(
            model_type = model_type,
            dataset = dataset,
            mobj = mobj,
            tr = tr,
            model_name = model_name,
            lr = lr,
            wd = wd,
            tries = tries,
            root_model = root_model,
            num_epochs = num_epochs,
            batch_size = batch_size,
            train_ratio = train_ratio,
            valid_ratio = valid_ratio,
            metal_ratio = metal_ratio,
        )
        if success:
            break
        k += 1
        if k > kmax:
            break

def exec_model(
    model_type,
    dataset,
    mobj,
    tr,
    model_name,
    lr = 1e-5,
    wd = 1e-7,
    tries = 1,
    root_model = 'c:/WORKSPACE_KRICT/MODELS/202204/nmre',
    num_epochs = 300,
    batch_size = 128,
    train_ratio = 0.7,
    valid_ratio = 0.2,
    metal_ratio = 1,
):
    def record(writer, tag, value, epoch):
        try:
            writer.add_scalar(tag, value, epoch)
        except:
            return
    gc.collect()
    torch.cuda.empty_cache()

    for n in range(0, tries):
        rseed  = 35 + n
        train_data, valid_data, test_data = dataset.train_test_split(train_ratio=train_ratio, 
                                                                     valid_ratio=valid_ratio,
                                                                     rseed=rseed,
                                                                     metal_ratio=metal_ratio)
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                                    collate_fn=tr.collate_fn)
        val_data_loader = DataLoader(valid_data, batch_size=batch_size, collate_fn=tr.collate_fn)
        test_data_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=tr.collate_fn)

        model = mobj(dataset.n_atom_feats, dataset.n_rdf_feature, dataset.n_bdf_feature).cuda()
        optimizer = AdaBound(model.parameters(), lr=lr, weight_decay=wd)
        criterion = torch.nn.L1Loss()

        root = os.path.join(root_model, model_type)
        if not os.path.isdir(root):
            os.makedirs(root)
        for i in range(99):            
            output_root = os.path.join(root, '{}_n{:02d}'.format(model_name, i))
            if not os.path.isdir(output_root) and not os.path.isfile(output_root):
                os.makedirs(output_root)
                break
            
        print(output_root)
        with open(os.path.join(output_root, 'params.json'),'w') as f:
            json.dump(dict(random_seed=rseed, learning_rate=lr, weight_decay=wd, 
                train_ratio=train_ratio, valid_ratio=valid_ratio, batch_size=batch_size,
                metal_ratio=metal_ratio), 
                f, indent=4)
        writer = SummaryWriter(output_root)
        #with torch.no_grad():
        #    dummy = iter(test_data_loader).next()
        #    writer.add_graph(model, dummy[:7])

        train_maes = list()
        for epoch in range(1, num_epochs+1):
            t1 = time.time()
            train_loss, train_mae = tr.train(model, optimizer, train_data_loader, criterion)
            valid_loss, valid_mae, valid_rmse, _, _, _ = tr.test(model, val_data_loader, criterion)
            train_maes.append(train_mae)
            if train_mae > 1 and epoch > 10 and np.std(train_maes[-3:]) < 1e-2:
                shutil.rmtree(output_root)
                return False
            if epoch%10 == 0:
                torch.save(model.state_dict(), 
                           os.path.join(output_root, 'model.{:05d}.pt'.format(epoch)))
                _, _, _, idxs, targets, preds = tr.test(model, test_data_loader, criterion)
                df = pd.DataFrame(dict(
                    id=idxs.astype(str).squeeze(), 
                    target=targets.squeeze(), 
                    prediction=preds.squeeze()))
                df = df.sort_values('id', key=lambda x: np.argsort(index_natsorted(df['id'])))
                df.to_csv(os.path.join(output_root, 'pred.{:05d}.csv'.format(epoch)), index=False)

            t = time.time() - t1

            record(writer, 'train/loss', train_loss, epoch)
            record(writer, 'train/MAE', train_mae, epoch)
#            record(writer, 'train/F1', train_f1, epoch)
            record(writer, 'valid/loss', valid_loss, epoch)
            record(writer, 'valid/MAE', valid_mae, epoch)
            record(writer, 'valid/RMSE', valid_rmse, epoch)
            record(writer, 'time/per_epoch', t, epoch)

            print('Epoch [{}/{}]\tTrain / Valid Loss: {:.4f} / {:.4f}\tMAE: {:.4f} / {:.4f} ({:.1f} sec)'
                    .format(epoch, num_epochs, train_loss, valid_loss, train_mae, valid_mae, t))
    return True


def test_model(
    dataset,
    output_root,
    mobj,
    tr,
    pfx='test.',
    batch_size=2048,
    model_order = -1,
    epoch_idx = -2
):
    gc.collect()
    torch.cuda.empty_cache()

    data_loader = DataLoader(dataset.crystal, batch_size=batch_size, collate_fn=tr.collate_fn)

    model = mobj(dataset.n_atom_feats, dataset.n_rdf_feature, dataset.n_bdf_feature)

    pts = []
    for pt in os.listdir(output_root):
        if not pt.endswith('.pt'): continue
        pts.append(pt)
    pts = sorted(pts)
    model_path = os.path.join(output_root, pts[model_order])
    epoch = int(pts[model_order].split('.')[epoch_idx])
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    _, _, _, idxs, targets, preds = tr.test(model, data_loader, None)
    df = pd.DataFrame(dict(
        id=idxs.astype(str).squeeze(), 
        target=targets.squeeze(), 
        prediction=preds.squeeze()))
    df = df.sort_values('id', key=lambda x: np.argsort(index_natsorted(df['id'])))
    df.to_csv(os.path.join(output_root, '{}{:05d}.csv'.format(pfx, epoch)), index=False)
