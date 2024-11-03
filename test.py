import os
import torch
import argparse
import numpy as np
import scipy.io as sio
from model.u2net import U2Net as Net
from utils.load_test_data import load_h5py_with_hp


def test(args):
    cut_size = args.cut_size
    pad = args.pad
    
    # load data
    model = Net(args.channels, args.spa_channels, args.spe_channels, H=cut_size + 2 * pad, W=cut_size + 2 * pad).to(args.device).eval()
    weight = torch.load(args.weight)

    try: 
        model.load_state_dict(weight)
    except:
        new_weight = {}
        for k, v in weight.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            new_weight[new_k] = v
        model.load_state_dict(new_weight)

    ms, _, pan, _ = load_h5py_with_hp(args.file_path)

    # get size
    image_num, C, h, w = ms.shape
    _, _, H, W = pan.shape
    ms_size = cut_size // 4
    edge_H = (cut_size - (H - (H // cut_size) * cut_size)) % cut_size
    edge_W = (cut_size - (W - (W // cut_size) * cut_size)) % cut_size

    for k in range(image_num):
        print('Processing the {}th sample...'.format(k+1))
        with torch.no_grad():
            x1, x2 = ms[k, :, :, :], pan[k, 0, :, :]
            x1 = x1.unsqueeze(dim=0).float().to(args.device)
            x2 = x2.unsqueeze(dim=0).unsqueeze(dim=1).float().to(args.device)
            x1_pad = torch.zeros(1, C, h + pad // 2 + edge_H // 4, w + pad // 2 + edge_W // 4).to(args.device)
            x2_pad = torch.zeros(1, 1, H + pad * 2 + edge_H, W + pad * 2 + edge_W).to(args.device)
            x1 = torch.nn.functional.pad(x1, (pad // 4, pad // 4, pad // 4, pad // 4), 'reflect')
            x2 = torch.nn.functional.pad(x2, (pad, pad, pad, pad), 'reflect')
            x1_pad[:, :, :h + pad // 2, :w + pad // 2] = x1
            x2_pad[:, :, :H + pad * 2, :W + pad * 2] = x2
            output = torch.zeros(1, C, H + edge_H, W + edge_W).to(args.device)

            scale_H = (H + edge_H) // cut_size
            scale_W = (W + edge_W) // cut_size
            for i in range(scale_H):
                for j in range(scale_W):
                    MS = x1_pad[:, :, i * ms_size: (i + 1) * ms_size + pad // 2,
                         j * ms_size: (j + 1) * ms_size + pad // 2]
                    PAN = x2_pad[:, :, i * cut_size: (i + 1) * cut_size + 2 * pad,
                          j * cut_size: (j + 1) * cut_size + 2 * pad]
                    sr = model(MS, PAN)
                    sr = torch.clamp(sr, 0, 1)
                    output[:, :, i * cut_size: (i + 1) * cut_size, j * cut_size: (j + 1) * cut_size] = \
                        sr[:, :, pad: cut_size + pad, pad: cut_size + pad] * 2047.
            output = output[:, :, :H, :W]
            output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_name = os.path.join(args.save_dir, "output_mulExm_" + str(k) + ".mat")
            sio.savemat(save_name, {'sr': output})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=32, help='Feature channels')
    parser.add_argument('--spa_channels', type=int, default=1, help='Feature channels')
    parser.add_argument('--spe_channels', type=int, default=8, help='Feature channels')
    parser.add_argument('--cut_size', type=int, default=256, help='cut size')
    parser.add_argument('--pad', type=int, default=0, help='padding size')
    parser.add_argument('--file_path', type=str, default='', help='Absolute path of the test file (in h5 format).')
    parser.add_argument('--save_dir', type=str, default='', help='Absolute path of the save dir.')
    parser.add_argument('--weight', type=str, default='weights/420.pth', help='Path of the weight.')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    test(args)
