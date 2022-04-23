# ------------------------------------------------------------------------
# Training code for bilinear similarity network (BMNet and BMNet+)
# --cfg: path for configuration file
# ------------------------------------------------------------------------
import argparse
import datetime
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import os 

from config import cfg 
import util.misc as utils
from loss import get_loss
from FSC147_dataset import build_dataset, batch_collate_fn 
from engine import evaluate, train_one_epoch, visualization 
from models import build_model

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def main(args):
    print(args)
    device = torch.device(cfg.TRAIN.device)
    # fix the seed for reproducibility
    seed = cfg.TRAIN.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(cfg)
    criterion = get_loss(cfg)
    criterion.to(device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.TRAIN.lr_backbone,
        },
    ]
    
    if cfg.TRAIN.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.lr,
                                      weight_decay=cfg.TRAIN.weight_decay)
    elif cfg.TRAIN.optimizer == "Adam":
        optimizer = torch.optim.Adam(param_dicts, lr=cfg.TRAIN.lr)
    elif cfg.TRAIN.optimizer == "SGD":
        optimizer = torch.optim.SGD(param_dicts, lr=cfg.TRAIN.lr,
                                    weight_decay=cfg.TRAIN.weight_decay)
    else:
        raise NotImplementedError
        
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.lr_drop)
    # define dataset
    dataset_train = build_dataset(cfg, is_train=True)
    dataset_val = build_dataset(cfg, is_train=False)

    data_loader_train = DataLoader(dataset_train, batch_size=cfg.TRAIN.batch_size, collate_fn=batch_collate_fn, shuffle=True, num_workers=cfg.TRAIN.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=batch_collate_fn)    
    output_dir = Path(cfg.DIR.output_dir)
    
    loss_list = []
    val_mae_list = []
    
    if cfg.VAL.evaluate_only:
        if os.path.isfile(cfg.VAL.resume):
            checkpoint = torch.load(cfg.VAL.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
        else:
            print('model state dict not found.')
        if cfg.VAL.visualization:
            mae = visualization(cfg, model, dataset_val, data_loader_val, device, cfg.DIR.output_dir)
        else:
            mae = evaluate(model, data_loader_val, device, cfg.DIR.output_dir)
        return
    
    if os.path.isfile(cfg.TRAIN.resume):
        if cfg.TRAIN.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.TRAIN.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.TRAIN.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not cfg.VAL.evaluate_only and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            cfg.TRAIN.start_epoch = checkpoint['epoch'] + 1
            loss_list = checkpoint['loss']
            val_mae_list = checkpoint['val_mae']

    best_mae = 10000 if len(val_mae_list) == 0 else min(val_mae_list)
    best_mae = 10000
    
    print("Start training")
    start_time = time.time()
    
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.epochs):
        loss = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            cfg.TRAIN.clip_max_norm)
        
        mae, mse = evaluate(model, data_loader_val, device, cfg.DIR.output_dir)
        loss_list.append(loss)
        val_mae_list.append(mae)
        lr_scheduler.step()
        if cfg.DIR.output_dir:
            checkpoint_path = os.path.join(cfg.DIR.output_dir, 'model_ckpt.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'config': cfg,
                'loss': loss_list,
                'val_mae': val_mae_list
            }, checkpoint_path) 
        
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            if cfg.DIR.output_dir:
                checkpoint_path = os.path.join(cfg.DIR.output_dir, 'model_best.pth')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'config': cfg,
                    'loss': loss_list,
                    'val_mae': val_mae_list
                }, checkpoint_path)
        
        utils.plot_learning_curves(loss_list, val_mae_list, cfg.DIR.output_dir)
        
        if cfg.DIR.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write('Epoch %d: loss %.8f, MAE %.2f, MSE %.2f, Best MAE %.2f, Best MSE %.2f \n'%(epoch +1, loss, mae, mse, best_mae, best_mse))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Class Agnostic Object Counting in PyTorch"
    )
    parser.add_argument(
        "--cfg",
        default="config/bmnet+_fsc147.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    #cfg.merge_from_list(args.opts)
    
    cfg.DIR.output_dir = os.path.join(cfg.DIR.snapshot, cfg.DIR.exp)
    if not os.path.exists(cfg.DIR.output_dir):
        os.mkdir(cfg.DIR.output_dir)    

    cfg.TRAIN.resume = os.path.join(cfg.DIR.output_dir, cfg.TRAIN.resume)
    cfg.VAL.resume = os.path.join(cfg.DIR.output_dir, cfg.VAL.resume)

    with open(os.path.join(cfg.DIR.output_dir, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    main(cfg)
