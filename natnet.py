from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

import models as mdl

import numpy as np
import argparse
import pickle
import os
import datetime
import time
import math
import shutil
from scipy.optimize import linear_sum_assignment
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])

class MyDataset(Dataset):
    def __init__(self):
        self.cifar10 = datasets.CIFAR10(root='.',download=True,train=True,transform=train_transform)

    def __getitem__(self, index):
        data, target = self.cifar10[index]

        # Your transformations here (or set it in CIFAR10)
                                            
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

def get_parser():
    parser = argparse.ArgumentParser(description='Scalable Compression')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')

    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')

    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to be used (acceptable values: cuda, cpu) (default: cuda)')

    parser.add_argument('--milestones', nargs="+", type=int, default=[30,60,80],
                        help='Milestones for learning rate decay (default: [30, 60, 80])')

    # not sure if I need
    parser.add_argument('--model', type=str, default='natnet',
                        help='model choise (acceptable values: rotnet, supervised, rot-nonlinear, rot-conv ) (default: rotnet)')

    parser.add_argument('--nins', type=int, default=4,
                        help='number of nin blocks to comprise the model (default: 4)')

    # not sure if I need
    parser.add_argument('--layer', type=int, default=2,
                        help='rotnet layer to take features from to use for classifier (default: 2)')

    parser.add_argument('--opt', type=str, default='sgd',
                        help='Optimizer to be used (acceptable values: sgd, adam) (default: sgd)')

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for optimizer (default: 0.1)')

    parser.add_argument('--weight_decay', default=5e-4, type=float)

    parser.add_argument('--print_after_batches', type=int, default=100,
                        help='Print training progress every print_after_batches batches (default: 2)')

    parser.add_argument('--results_dir', default='results/', type=str)

    parser.add_argument('--suffix', default='', type=str, 
                        help="When I need to custom name the final results folder, must begin with _")

    parser.add_argument('--epochs_to_save', nargs="+", type=int, default=[100],
                        help='List of epochs to save (default: [100])')

    return parser 

parser = get_parser()
args = parser.parse_args()

out_size = 10
device = args.device

c = torch.from_numpy(np.random.normal(0, 1, [50000, out_size]).astype(np.float32)).to(device)
cnorm = torch.linalg.norm(c, dim=1).reshape(-1,1)
c = c/cnorm
p = torch.from_numpy(np.zeros([50000,1]).astype(np.int)).to(device)
for i in range(p.size()[0]):
    p[i] = i

def train(args, network, train_loader, optimizer, mult, scheduler, epoch):
    network.train()
    total_images_till_now = 0
    total_images = len(train_loader.dataset)*mult
    for batch_idx, (data, target, indx) in enumerate(train_loader):
        data, target, indx = Variable(data).to(device), Variable(target).to(device), Variable(indx).to(device)
        optimizer.zero_grad()
        output, _, _ = network(data)
        output = output/torch.linalg.norm(output)

        pbatch = p[indx]

        if epoch % 3 == 0:
            cost = np.zeros([output.size()[0], output.size()[0]]).astype(np.float32)
            for i in range(output.size()[0]):
                for j in range(output.size()[1]):
                    cost[i][j] = torch.linalg.norm(output[i]-c[p[indx[j]]])
                    optind = linear_sum_assignment(cost)
                    for i in range(len(optind)):
                        p[indx[optind[i][0]]] = pbatch[optind[i][1]]
                        
        y = c[p[indx]].reshape(-1,out_size).to(device)
        lost = torch.nn.MSELoss()
        loss = lost(output, y)

        loss.backward()
        optimizer.step()
        total_images_till_now = total_images_till_now + len(data)
        if batch_idx % args.print_after_batches == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, total_images_till_now, total_images,
                100. * total_images_till_now/total_images, loss.item()))

    scheduler.step()

    return


def test(args, network, test_loader, mult, datatype):
    network.eval()
    test_loss = 0
    correct = 0
    for data,target in test_loader:
        data, target = data.to(args.device), target.to(device)
        output, _, _ = network(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    total_images = len(test_loader.dataset)*mult
    test_loss /= total_images
    test_acc = 100. * correct / total_images
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        datatype, test_loss, correct, total_images, test_acc))

    return test_loss, test_acc
    



def main(args):
    # hard coded values
    in_channels = 3 # rgb channels of input image
    out_classes = out_size # d length
    lr_decay_rate = 0.2 # lr is multiplied by decay rate after a milestone epoch is reached
    mult = 1 # data become mult times 
    ####################

    #train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.ToTensor()
        
    trainset = MyDataset()
    testset = datasets.CIFAR10(root='.', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    network = mdl.RotNet(in_channels=in_channels, num_nin_blocks=args.nins, out_classes=out_classes).to(args.device) 

    if args.opt == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=lr_decay_rate)

    ####################################### Saving information
    results_dict = {}
    # These will store the values for best test accuracy model
    results_dict['train_loss'] = -1
    results_dict['train_acc'] = -1
    results_dict['test_loss'] = -1
    results_dict['test_acc'] = -1
    results_dict['best_acc_epoch'] = -1
    # For storing training history
    results_dict['train_loss_hist'] = []
    results_dict['train_acc_hist'] = []
    results_dict['test_loss_hist'] = []
    results_dict['test_acc_hist'] = []

    # directories to save models
    checkpoint_path = os.path.join(args.results_dir, 'model.pth')
    checkpoint_path_best_acc = os.path.join(args.results_dir, 'model_best_acc.pth')

    test_acc_max = -math.inf
    loop_start_time = time.time()
    checkpoint = {}
    for epoch in range(args.epochs):
        train(args, network, train_loader, optimizer, mult, scheduler, epoch)
        
        train_loss, train_acc = test(args, network, test_loader, mult, 'Train')
        results_dict['train_loss_hist'].append(train_loss)
        results_dict['train_acc_hist'].append(train_acc)
        
        test_loss, test_acc = test(args, network, test_loader, mult, 'Test')
        results_dict['test_loss_hist'].append(test_loss)
        results_dict['test_acc_hist'].append(test_acc)
        print('Epoch {} finished --------------------------------------------------------------------------', epoch+1)
        
        checkpoint = {'model_state_dict': network.state_dict(), 
                      'optimizer_state_dict': optimizer.state_dict(), 
                      'epoch':epoch+1,  
                      'train_loss':train_loss, 
                      'train_acc':train_acc, 
                      'test_loss':test_loss, 
                      'test_acc':test_acc}

        if test_acc > test_acc_max:
            test_acc_max = test_acc
            if os.path.isfile(checkpoint_path_best_acc):
                os.remove(checkpoint_path_best_acc)

            torch.save(checkpoint, checkpoint_path_best_acc)
            
            results_dict['best_acc_epoch'] = epoch+1
            results_dict['train_loss'] = train_loss
            results_dict['train_acc'] = train_acc
            results_dict['test_loss'] = test_loss
            results_dict['test_acc'] = test_acc

        if epoch+1 in args.epochs_to_save:
            torch.save(checkpoint, os.path.join(args.results_dir, 'model_epoch_'+str(epoch+1)+'.pth'))


    torch.save(checkpoint, checkpoint_path)
        
    print('Total time for training loop = ', time.time()-loop_start_time)

    return results_dict



# Starting the program execution from here
if __name__ == '__main__':
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    args.results_dir = os.path.join(args.results_dir, 'natnet_'+str(args.nins)+'_ninblocks'+args.suffix)

    assert (not os.path.exists(args.results_dir))

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    results_file = os.path.join(args.results_dir, 'results_dict.pickle')

    print('--------------------------------------------------------')
    print('--------------------------------------------------------')
    print('Experiment starting at ', datetime.datetime.now())
    print(' ')
    options = vars(args)
    keys = options.keys()
    for key in keys:
        print(key, ': ', options[key])
    print(' ')
    print('--------------------------------------------------------')
    print('--------------------------------------------------------')
    print(' ')
    print(' ')

    results_dict = main(args)
    
    # saving the configuration 
    for key in keys:
        new_key = 'config_' + key
        results_dict[new_key] = options[key]

    with open(results_file, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print('--------------------------------------------------------')
    print('--------------------------------------------------------')
    print('Total time for experiment: ', time.time()-start_time, ' seconds')
    print('--------------------------------------------------------')
    print('--------------------------------------------------------')  

