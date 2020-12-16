import torch.nn as nn
from collections import OrderedDict

class NIN_block(nn.Module):
    def __init__(self, in_ch1, out_ch1, out_ch2, out_ch3, ksize):
        super(NIN_block, self).__init__()
        
        pad = int((ksize-1)/2)
        
        self.block = nn.Sequential(OrderedDict([
                ('main_conv', nn.Conv2d(in_ch1, out_ch1, ksize, padding=pad)),
                ('main_bnorm', nn.BatchNorm2d(out_ch1)),
                ('main_relu', nn.ReLU()), 
                ##### 
                ('mlp_conv1', nn.Conv2d(out_ch1, out_ch2, 1)),
                ('mlp_bnorm1', nn.BatchNorm2d(out_ch2)),
                ('mlp_relu1', nn.ReLU()), 
                #####
                ('mlp_conv2', nn.Conv2d(out_ch2, out_ch3, 1)),
                ('mlp_bnorm2', nn.BatchNorm2d(out_ch3)),
                ('mlp_relu2', nn.ReLU())]))

    def forward(self, x):
        x = self.block(x)
        return x


class RotNet(nn.Module):
    def __init__(self, in_channels, num_nin_blocks, out_classes):
        super(RotNet, self).__init__()
        
        self.rotnet_layer_names = []
        for i in range(num_nin_blocks):
            self.rotnet_layer_names.append('nin_block'+str(i+1))
            if i < 2:
                self.rotnet_layer_names.append('maxpool'+str(i+1))
        
        self.rotnet = nn.ModuleDict({})

        self.rotnet['nin_block1'] = NIN_block(in_channels, 192, 160, 96, 5)
        self.rotnet['maxpool1'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rotnet['nin_block2'] = NIN_block(96, 192, 192, 192, 5)
        self.rotnet['maxpool2'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        assert num_nin_blocks > 2
        for i in range(2,num_nin_blocks):
            nin_block_name = 'nin_block' + str(i+1)
            out_ch3 = 192
            # code below can be uncommented to make this model exactly like in NiN paper
            # after removing the fully connected layer as well
            # if i+1 == num_nin_blocks:
            #     out_ch3 = out_classes
            self.rotnet[nin_block_name] = NIN_block(192, 192, 192, out_ch3, 3)

        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(192, out_classes)


    def forward(self, x, ret_block_nums=[]):
        # ret_blocks: list of numbers where 
        # each number corresponds to nin block number
        # who ouput you want in features
        ret_layer_names = []
        for i in ret_block_nums:
            if i < 3:
                ret_layer_names.append('maxpool'+str(i))
            else:
                ret_layer_names.append('nin_block'+str(i))
        
        ret_feats = {}
        for layer in self.rotnet_layer_names:
            x = self.rotnet[layer](x)
            if layer in ret_layer_names:
                ret_feats[layer] = x

        x = self.glob_avg_pool(x)
        x = self.flat(x)
        x = self.fc(x)
        
        ret_block_num2names = dict(zip(ret_block_nums, ret_layer_names)) 

        return x, ret_feats, ret_block_num2names
        

class ConvClassifier(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(ConvClassifier, self).__init__()
        
        self.block = nn.Sequential(OrderedDict([
                ('nin_block3', NIN_block(in_channels, 192, 192, 192, 3)),
                ('glob_avg_pool', nn.AdaptiveAvgPool2d((1,1))),
                ('flat', nn.Flatten()),
                ('fc', nn.Linear(192, out_classes))]))

    def forward(self, x):
        x = self.block(x)
        return x


class NonLinearClassifier(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(NonLinearClassifier, self).__init__()

        self.block = nn.Sequential(OrderedDict([
                #('flat', nn.Flatten()),
                ('fc1', nn.Linear(in_channels, 200)),
                ('fc1_bnorm', nn.BatchNorm1d(200)),
                ('fc1_relu', nn.ReLU()), 
                ##### 
                ('fc2', nn.Linear(200, 200)),
                ('fc2_bnorm', nn.BatchNorm1d(200)),
                ('fc2_relu', nn.ReLU()),
                #####
                ('fc3', nn.Linear(200, out_classes))]))

    def forward(self, x):
        x = self.block(x)
        return x

'''
import torch
network = RotNet(3,5,4)
a = torch.randn(1,3,32,32)
ret_nums = [1,2,3,4,5]
out, feats, num2names = network(a, ret_nums)
print(out.shape)
for i in range(5):
    print(feats[num2names[ret_nums[i]]].shape)

torch.Size([1, 4])
torch.Size([1, 96, 16, 16])
torch.Size([1, 192, 8, 8])
torch.Size([1, 192, 8, 8])
torch.Size([1, 192, 8, 8])
torch.Size([1, 192, 8, 8])
'''
