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

def get_parser():
    parser = argparse.ArgumentParser(description='Supervised Conv Classifier')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')

    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')

    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to be used (acceptable values: cuda, cpu) (default: cuda)')

    parser.add_argument('--milestones', nargs="+", type=int, default=[35, 70, 85, 100],
                        help='Milestones for learning rate decay (default: [20,40,45,50])')

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

    parser.add_argument('--suffix_rot', default='', type=str, 
                        help="When I need to load a custom named rotnet model")

    parser.add_argument('--rot_model_type', default='', type=str, 
                        help="Which rotation model to load. Options: '', '_best_acc', '_epoch_100' (default: '') ")

    return parser 


def train(args, rot_network, class_network, train_loader, rot_optimizer, class_optimizer, mult, rot_scheduler, class_scheduler, epoch, in_features):
    rot_network.train()
    class_network.train()
    total_images_till_now = 0
    total_images = len(train_loader.dataset)*mult
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(args.device)
        target = target.to(args.device)
        rot_optimizer.zero_grad()
        class_optimizer.zero_grad()
        _, out_dict, layer_num2name_dict = rot_network(data, [args.layer])
        output = class_network(out_dict[layer_num2name_dict[args.layer]])
        loss = F.cross_entropy(output, target)
        loss.backward()
        rot_optimizer.step()
        class_optimizer.step()
        total_images_till_now = total_images_till_now + len(data)
        if batch_idx % args.print_after_batches == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, total_images_till_now, total_images,
                100. * total_images_till_now/total_images, loss.item()))
                

    rot_scheduler.step()
    class_scheduler.step()

    return


def test(args, rot_network, class_network, test_loader, mult, datatype, in_features):
    rot_network.eval()
    class_network.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(args.device)
        target = target.to(args.device)
        _, out_dict, layer_num2name_dict = rot_network(data, [args.layer])
        output = class_network(out_dict[layer_num2name_dict[args.layer]])
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
    in_channels = 3 # rgb channels of orignal image fed to rotnet
    if args.layer == 1:
        in_features = 96
    else:
        in_features = 192
    rot_classes = 4
    out_classes = 10 
    lr_decay_rate = 0.2 # lr is multiplied by decay rate after a milestone epoch is reached
    mult = 1 # data become mult times 
    ####################

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
                                            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255, 66.7/255))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255, 66.7/255))])
        
    trainset = datasets.CIFAR10(root='results/', train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root='results/', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rot_network = mdl.RotNet(in_channels=in_channels, num_nin_blocks=args.nins, out_classes=rot_classes).to(args.device) 
    class_network = mdl.ConvClassifier(in_channels=in_features, out_classes=out_classes).to(args.device)

    if args.opt == 'adam':
        rot_optimizer = optim.Adam(rot_network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        class_optimizer = optim.Adam(class_network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        rot_optimizer = optim.SGD(rot_network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        class_optimizer = optim.SGD(class_network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  

    rot_scheduler = optim.lr_scheduler.MultiStepLR(rot_optimizer, milestones=args.milestones, gamma=lr_decay_rate)
    class_scheduler = optim.lr_scheduler.MultiStepLR(class_optimizer, milestones=args.milestones, gamma=lr_decay_rate)

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


    #########
    test_acc_max = -math.inf
    loop_start_time = time.time()
    checkpoint = {}
    for epoch in range(args.epochs):
        train(args, rot_network, class_network, train_loader, rot_optimizer, class_optimizer, mult, rot_scheduler, class_scheduler, epoch, in_features)
        
        train_loss, train_acc = test(args, rot_network, class_network, train_loader, mult, 'Train', in_features)
        results_dict['train_loss_hist'].append(train_loss)
        results_dict['train_acc_hist'].append(train_acc)
        
        test_loss, test_acc = test(args, rot_network, class_network, test_loader, mult, 'Test', in_features)
        results_dict['test_loss_hist'].append(test_loss)
        results_dict['test_acc_hist'].append(test_acc)
        print('Epoch {} finished --------------------------------------------------------------------------'.format(epoch+1))
        
        checkpoint = {'class_model_state_dict': class_network.state_dict(), 
                      'class_optimizer_state_dict': class_optimizer.state_dict(), 
                      'rot_model_state_dict': rot_network.state_dict(), 
                      'rot_optimizer_state_dict': rot_optimizer.state_dict(), 
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


    torch.save(checkpoint, checkpoint_path)
        
    print('Total time for training loop = ', time.time()-loop_start_time)

    return results_dict



# Starting the program execution from here
if __name__ == '__main__':
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    assert (args.layer >= 1 and args.layer <=5)

    args.results_dir = os.path.join(args.results_dir, 'super_conv_classifier', 'nins_'+str(args.nins)+'_layer_'+str(args.layer)+args.suffix)

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

