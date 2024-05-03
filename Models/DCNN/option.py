import argparse

# Training settings
parser = argparse.ArgumentParser(description="Super-Resolution")
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument("--nEpochs", type=int, default=200, help="maximum number of epochs to train")  # 200
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--lr", type=int, default=1e-4, help="lerning rate")
parser.add_argument("--lrf", type=int, default=1e-3, help="lerning rate")
parser.add_argument("--cuda", type=str, default="0", help="Use cuda")
parser.add_argument("--threads", type=int, default=20, help="number of threads for dataloader to use")
parser.add_argument("--resume",
                    default="",
                    type=str,
                    help="Path to checkpoint (default: none)  checkpoint/model_epoch_95.pth")
# Network settings
parser.add_argument('--n_subs', type=int, default=8, help='暂定')
parser.add_argument('--n_ovls', type=int, default=2, help='暂定')
# Test image
# parser.add_argument('--method', default='DCNN', type=str, help='super resolution method name')
##############################################################################
parser.add_argument("--datasetName", default="Cave", type=str, help="data name")
parser.add_argument('--save_name', default='cave_x2_dcnn',
                    type=str, help='')
parser.add_argument('--scale', type=int, default=2, help='暂定')
parser.add_argument('--n_colors', type=int, default=480, help='暂定')  # 31
parser.add_argument('--n_feats', type=int, default=64, help='暂定')  # 64
parser.add_argument('--n_latent', type=int, default=3, help='暂定')  # 3
parser.add_argument('--train_path', default='./CAVE/train', type=str, help='')
parser.add_argument('--test_path', default='./CAVE/test', type=str, help='')
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")  # 16
parser.add_argument("--patch_size", type=int, default=32, help="training patch size")  # 32
parser.add_argument("--gpus", default="0,1", type=str, help="gpu ids (default: 0)")
##############################################################################
parser.add_argument('--model_name', default='./checkpoint/DCNN_cave_x2/epoch_200.pth', type=str, help='test_path')
opt = parser.parse_args()
