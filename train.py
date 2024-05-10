import os
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.tools import ERGAS
from torchinfo import summary
import torch.distributed as dist
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from model.u2net import U2Net as Net
from torch.utils.data import DataLoader
from utils.load_train_data import Dataset_Pro


SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True


def save_checkpoint(args, model, epoch):
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    model_out_path = args.weight_dir + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)


def prepare_training_data(args):
    train_set = Dataset_Pro(args.train_data_path)
    validate_set = Dataset_Pro(args.val_data_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=args.batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=args.batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True)
    return training_data_loader, validate_data_loader


def train(args, training_data_loader, validate_data_loader):
    model = Net(args.channels, args.spa_channels, args.spe_channels, args.H, args.W, args.ratio).to(args.device)
    summary(model, input_size=[(args.batch_size, args.spe_channels, 16, 16),
                               (args.batch_size, args.spa_channels, 64, 64)],
            dtypes=[torch.float, torch.float])

    if args.use_ergas is True:
        criterion0 = nn.L1Loss(size_average=True).to(args.device)
        criterion1 = ERGAS(args.ratio).to(args.device)
    else:
        criterion = nn.L1Loss(size_average=True).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step, gamma=args.decay)

    t_start = time.time()
    print('Start training...')

    # train
    for epoch in range(0, args.epoch, 1):
        epoch += 1
        model.train()
        epoch_train_loss = []
        epoch_train_loss0 = []
        epoch_train_loss1 = []
        for iteration, batch in enumerate(training_data_loader, 1):
            gt, pan, ms = batch[0].to(args.device), batch[3].to(args.device), batch[4].to(args.device)
            optimizer.zero_grad()
            sr = model(ms, pan)
            if args.use_ergas is True:
                loss0 = criterion0(sr, gt)
                loss1 = criterion1(sr, gt)
                loss = loss0 + args.ergas_hp * loss1
                epoch_train_loss0.append(loss0.item())
                epoch_train_loss1.append(loss1.item())
            else:
                loss = criterion(sr, gt)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        if args.use_ergas is True:
            print('Epoch: {}/{}  training loss: {:.7f}  l1: {:.7f}  ergas: {:.7f}'
                  .format(epoch, args.epoch, t_loss, np.nanmean(np.array(epoch_train_loss0)),
                          np.nanmean(np.array(epoch_train_loss1))))
        else:
            print('Epoch: {}/{}  training loss: {:.7f}'.format(epoch, args.epoch, t_loss))

        # validate
        with torch.no_grad():
            if epoch % 10 == 0:
                model.eval()
                epoch_val_loss = []
                epoch_val_loss0 = []
                epoch_val_loss1 = []
                for iteration, batch in enumerate(validate_data_loader, 1):
                    gt, pan, ms = batch[0].to(args.device), batch[3].to(args.device), batch[4].to(args.device)
                    sr = model(ms, pan)
                    if args.use_ergas is True:
                        loss0 = criterion0(sr, gt)
                        loss1 = criterion1(sr, gt)
                        loss = loss0 + args.ergas_hp * loss1
                        epoch_val_loss0.append(loss0.item())
                        epoch_val_loss1.append(loss1.item())
                    else:
                        loss = criterion(sr, gt)
                    epoch_val_loss.append(loss.item())
                v_loss = np.nanmean(np.array(epoch_val_loss))
                t_end = time.time()
                if args.use_ergas is True:
                    print('---------------validate loss: {:.7f}  l1: {:.7f} ergas: {:.7f}----------------'
                          .format(v_loss, np.nanmean(np.array(epoch_val_loss0)), np.nanmean(np.array(epoch_val_loss1))))
                else:
                    print('---------------validate loss: {:.7f}---------------'.format(v_loss))
                print('-----------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
                t_start = time.time()

        # save weights
        if epoch % args.ckpt == 0:
            save_checkpoint(args, model, epoch)
        else:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=int, default=4, help='Upsample ratio')
    parser.add_argument('--H', type=int, default=64, help='Height of the high-resolution image')
    parser.add_argument('--W', type=int, default=64, help='Width of the high-resolution image')
    parser.add_argument('--channels', type=int, default=32, help='Feature channels')
    parser.add_argument('--spa_channels', type=int, default=1, help='Spatial channels')
    parser.add_argument('--spe_channels', type=int, default=8, help='Spectral channels')
    parser.add_argument('--use_ergas', type=bool, default=False, help='Use ERGAS loss for training or not')
    parser.add_argument('--ergas_hp', type=float, default=1e-4, help='Hyper-parameter for the ERGAS loss')
    parser.add_argument('--epoch', type=int, default=400, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--step', type=int, default=200, help='Step number')
    parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay')
    parser.add_argument('--ckpt', type=int, default=50, help='Checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_data_path', type=str, default='', help='Path of the training dataset.')
    parser.add_argument('--val_data_path', type=str, default='', help='Path of the validation dataset.')
    parser.add_argument('--weight_dir', type=str, default='weights/', help='Dir of the weight.')
    args = parser.parse_args()

    training_data_loader, validate_data_loader = prepare_training_data(args)
    train(args, training_data_loader, validate_data_loader)