from __future__ import print_function
import argparse
import torch

import sys
sys.path.append('./model')
sys.path.append('./datasets')
sys.path.append('./metric');
from solver_MSDA import Solver
import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MSDA Implementation')
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--record_folder', type=str, default='record', metavar='N',
                    help='record folder')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                    help='how many epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--source', type=str, default='svhn', metavar='N',
                    help='source dataset')
parser.add_argument('--target', type=str, default='mnist', metavar='N', help='target dataset')
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


def main():
    # if not args.one_step:

    solver = Solver(args, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, 
                    checkpoint_dir=args.checkpoint_dir,
                    save_epoch=args.save_epoch)
    record_num = 0
    
    record_train = '%s/%s_%s.txt' % (
        args.record_folder, args.target,record_num)
    record_test = '%s/%s_%s_test.txt' % (
        args.record_folder, args.target, record_num)
    while os.path.exists(record_train):
        record_num += 1
        record_train = '%s/%s_%s.txt' % (
            args.record_folder, args.target, record_num)
        record_test = '%s/%s_%s_test.txt' % (
            args.record_folder, args.target, record_num)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.record_folder):
        os.mkdir(args.record_folder.split('/')[0])
        os.mkdir(args.record_folder)
    if args.eval_only:
        solver.test(0)
    else:
        count = 0
        for t in range(args.max_epoch):
            print(t)
            if not args.one_step:
                # num = solver.train_merge_baseline(t, record_file=record_train)
                num = solver.train_MSDA(t, record_file=record_train)
            else:
                num = solver.train_onestep(t, record_file=record_train)
            count += num
            if t % 1 == 0:
                solver.test(t, record_file=record_test, save_model=args.save_model)
            if count >= 20000:
                break


if __name__ == '__main__':
    main()
