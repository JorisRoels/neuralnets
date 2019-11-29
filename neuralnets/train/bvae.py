"""
    This is a script that illustrates training a U-Net
"""

"""
    Necessary libraries
"""
import argparse
import datetime
import os

import torch.optim as optim
from torch.utils.data import DataLoader

from neuralnets.data.datasets import StronglyLabeledVolumeDataset
from neuralnets.networks.bvae import BVAE
from neuralnets.util.losses import CrossEntropyLoss, KLDLoss
from neuralnets.util.preprocessing import get_augmenter2d
from neuralnets.util.tools import set_seed

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()

# logging parameters
parser.add_argument("--seed", help="Seed for randomization", type=int, default=0)
parser.add_argument("--log_dir", help="Logging directory", type=str, default="bvae")
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses",
                    type=int, default=50)

# network parameters
parser.add_argument("--data_dir", help="Data directory", type=str, default="../../../data")
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network",
                    type=str, default="256,256")
parser.add_argument("--fm", help="Number of initial feature maps in the segmentation U-Net", type=int, default=64)
parser.add_argument("--levels", help="Number of levels in the segmentation U-Net (i.e. number of pooling stages)",
                    type=int, default=5)
parser.add_argument("--dropout", help="Dropout", type=float, default=0.0)
parser.add_argument("--norm", help="Normalization in the network (batch or instance)", type=str, default="batch")
parser.add_argument("--activation", help="Non-linear activations in the network", type=str, default="relu")
parser.add_argument("--beta", help="Beta value for the KL loss", type=float, default=0)
parser.add_argument("--bottleneck", help="Dimensionality of the bottleneck", type=int, default=16)

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay",
                    type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=200)
parser.add_argument("--len_epoch", help="Number of iteration in each epoch", type=int, default=1000)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=4)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=4)

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]
loss_rec_fn = CrossEntropyLoss()
loss_kl_fn = KLDLoss()

"""
Fix seed (for reproducibility)
"""
set_seed(args.seed)

"""
    Setup logging directory
"""
print('[%s] Setting up log directories' % (datetime.datetime.now()))
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

"""
    Load the data
"""
input_shape = (1, 2*args.input_size[0], 2*args.input_size[1])
print('[%s] Loading data' % (datetime.datetime.now()))
augmenter = get_augmenter2d(crop_shape=[args.input_size[0], args.input_size[1]],
                            sample_shape=[2*args.input_size[0], 2*args.input_size[1]], include_segmentation=False)
train = StronglyLabeledVolumeDataset(os.path.join(args.data_dir, 'EM/EPFL/training.tif'),
                                     os.path.join(args.data_dir, 'EM/EPFL/training_groundtruth.tif'),
                                     input_shape=input_shape, len_epoch=args.len_epoch)
test = StronglyLabeledVolumeDataset(os.path.join(args.data_dir, 'EM/EPFL/testing.tif'),
                                    os.path.join(args.data_dir, 'EM/EPFL/testing_groundtruth.tif'),
                                    input_shape=input_shape, len_epoch=args.len_epoch)
train_loader = DataLoader(train, batch_size=args.train_batch_size)
test_loader = DataLoader(test, batch_size=args.train_batch_size)

"""
    Build the network
"""
print('[%s] Building the network' % (datetime.datetime.now()))
net = BVAE(beta=args.beta, input_size=args.input_size[0]//2, bottleneck_dim=args.bottleneck, feature_maps=args.fm, levels=args.levels,
           dropout_enc=args.dropout, dropout_dec=args.dropout, norm=args.norm, activation=args.activation)

"""
    Setup optimization for training
"""
print('[%s] Setting up optimization for training' % (datetime.datetime.now()))
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Train the network
"""
print('[%s] Starting training' % (datetime.datetime.now()))
net.train_net(train_loader, test_loader, loss_rec_fn, loss_kl_fn, optimizer, args.epochs, scheduler=scheduler,
              augmenter=augmenter, print_stats=args.print_stats, log_dir=args.log_dir)

# """
#     Validate the trained network
# """
# validate(net, test.data, test.labels, args.input_size, batch_size=args.test_batch_size,
#          write_dir=os.path.join(args.write_dir, 'segmentation_final'),
#          val_file=os.path.join(args.log_dir, 'validation_final.npy'))
# net = torch.load(os.path.join(args.log_dir, 'best_checkpoint.pytorch'))
# validate(net, test.data, test.labels, args.input_size, batch_size=args.test_batch_size,
#          write_dir=os.path.join(args.write_dir, 'segmentation_best'),
#          val_file=os.path.join(args.log_dir, 'validation_best.npy'))

print('[%s] Finished!' % (datetime.datetime.now()))
