import torch
import os 
import math
import random
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class EqualizedConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, bias=True):
        super(EqualizedConv2d, self).__init__()
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.weight_param = nn.Parameter(torch.FloatTensor(out_features, in_features, kernel_size, kernel_size).normal_(0.0, 1.0))
        if self.bias:
            self.bias_param = nn.Parameter(torch.FloatTensor(out_features).fill_(0))
        fan_in = kernel_size * kernel_size * in_features
        self.scale = math.sqrt(2. / fan_in)
    def forward(self, x):
        return F.conv2d(input=x,
                        weight=self.weight_param.mul(self.scale),  # scale the weight on runtime
                        bias=self.bias_param if self.bias else None,
                        stride=self.stride, padding=self.padding)

class EqualizedDeconv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, bias=True):
        super(EqualizedDeconv2d, self).__init__()
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.weight_param = nn.Parameter(torch.FloatTensor(in_features, out_features, kernel_size, kernel_size).normal_(0.0, 1.0))
        if self.bias:
            self.bias_param = nn.Parameter(torch.FloatTensor(out_features).fill_(0))
        fan_in = in_features
        self.scale = math.sqrt(2. / fan_in)
    def forward(self, x):
        return F.conv_transpose2d(input=x,
                                  weight=self.weight_param.mul(self.scale),  # scale the weight on runtime
                                  bias=self.bias_param if self.bias else None,
                                  stride=self.stride, padding=self.padding)

class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(EqualizedLinear, self).__init__()
        self.bias = bias
        self.weight_param = nn.Parameter(torch.FloatTensor(out_features, in_features).normal_(0.0, 1.0))
        if self.bias:
            self.bias_param = nn.Parameter(torch.FloatTensor(out_features).fill_(0))
        fan_in = in_features
        self.scale = math.sqrt(2. / fan_in)
    def forward(self, x):
        N = x.size(0)
        return F.linear(input=x.view(N,-1), weight=self.weight_param.mul(self.scale),
                        bias=self.bias_param if self.bias else None)

#----------------------------------------------------------------------------
# Minibatch standard deviation.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L127

class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()
    def forward(self, x):
        y = x - torch.mean(x, dim=0, keepdim=True)       # [NCHW] Subtract mean over batch.
        y = torch.mean(y.pow(2.), dim=0, keepdim=False)  # [CHW]  Calc variance over batch.
        y = torch.sqrt(y + 1e-8)                         # [CHW]  Calc stddev over batch.
        y = torch.mean(y).view(1,1,1,1)                  # [1111] Take average over fmaps and pixels.
        y = y.repeat(x.size(0),1,x.size(2),x.size(3))    # [N1HW] Replicate over batch and pixels.
        return torch.cat([x, y], 1)                      # [N(C+1)HW] Append as new fmap.

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120

class PixelwiseNorm(nn.Module):
    def __init__(self, sigma=1e-8):
        super(PixelwiseNorm, self).__init__()
        self.sigma = sigma # small number for numerical stability
    def forward(self, x):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(self.sigma).sqrt() # [N1HW]
        return x.div(y)

#----------------------------------------------------------------------------
# Smoothly fade in the new layers.

class ConcatTable(nn.Module):
    def __init__(self, layer1, layer2):
        super(ConcatTable, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2
    def forward(self,x):
        return [self.layer1(x), self.layer2(x)]

class Fadein(nn.Module):
    def __init__(self, alpha=0.):
        super(Fadein, self).__init__()
        self.alpha = alpha
    def update_alpha(self, delta):
        self.alpha = self.alpha + delta
        self.alpha = max(0, min(self.alpha, 1.0))
    def get_alpha(self):
        return self.alpha
    def forward(self, x):
        # x is a ConcatTable, with x[0] being old layer, x[1] being the new layer to be faded in
        return x[0].mul(1.0-self.alpha) + x[1].mul(self.alpha)

#----------------------------------------------------------------------------
# Nearest-neighbor upsample
# define this myself because torch.nn.Upsample has been deprecated

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

def conv_block(layers, in_features, out_features, kernel_size, stride, padding, pixel_norm):
    layers.append(EqualizedConv2d(in_features, out_features, kernel_size, stride, padding))
    layers.append(nn.LeakyReLU(0.2))
    if pixel_norm:
        layers.append(PixelwiseNorm())
    return layers

def deconv_block(layers, in_features, out_features, kernel_size, stride, padding, pixel_norm):
    layers.append(EqualizedDeconv2d(in_features, out_features, kernel_size, stride, padding))
    layers.append(nn.LeakyReLU(0.2))
    if pixel_norm:
        layers.append(PixelwiseNorm())
    return layers

def deepcopy_layers(module, layer_name):
    # copy the layer with name in "layer_name"
    new_module = nn.Sequential()
    for name, m in module.named_children():
        if name in layer_name:
            new_module.add_module(name, m)                 # construct new structure
            new_module[-1].load_state_dict(m.state_dict()) # copy weights
    return new_module

def deepcopy_exclude(module, exclude_name):
    # copy all the layers EXCEPT "layer_name"
    new_module = nn.Sequential()
    for name, m in module.named_children():
        if name not in exclude_name:
            new_module.add_module(name, m)                 # construct new structure
            new_module[-1].load_state_dict(m.state_dict()) # copy weights
    return new_module


#----------------------------------------------------------------------------
# generator

class Generator(nn.Module):
    def __init__(self, nc=3, nz=512, size=256, cond=False, num_classes=7):
        super(Generator, self).__init__()
        self.nc = nc # number of channels of the generated image
        self.nz = nz # dimension of the input noise
        self.size = size # the final size of the generated image
        self.cond = cond # true: conditional gan
        self.num_classes = num_classes
        self.stages = int(math.log2(self.size/4)) + 1 # the total number of stages (7 when size=256)
        self.current_stage = 1
        self.nf = lambda stage: min(int(8192 / (2.0 ** stage)), self.nz) # the number of channels in a particular stage
        self.module_names = []
        self.model = self.get_init_G()
        # one-hot embedding
        self.embedding = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.num_classes)
        nn.init.eye_(self.embedding.weight)
        self.embedding.weight.requires_grad_(False)
    def get_init_G(self):
        model = nn.Sequential()
        model.add_module('stage_{}'.format(self.current_stage), self.first_block())
        model.add_module('to_rgb', self.to_rgb_block(self.nf(self.current_stage)))
        return model
    def first_block(self):
        layers = []
        ndim = self.nf(self.current_stage)
        if not self.cond:
            layers.append(PixelwiseNorm()) # normalize latent vectors before feeding them to the network
        layers = deconv_block(layers, in_features=self.nz+self.num_classes if self.cond else self.nz,
            out_features=ndim, kernel_size=4, stride=1, padding=0, pixel_norm=True)
        layers = conv_block(layers, in_features=ndim, out_features=ndim, kernel_size=3, stride=1, padding=1, pixel_norm=True)
        return  nn.Sequential(*layers)
    def to_rgb_block(self, ndim):
        return EqualizedConv2d(in_features=ndim, out_features=self.nc, kernel_size=1, stride=1, padding=0)
    def intermediate_block(self, stage):
        assert stage > 1, 'For intermediate blocks, stage should be larger than 1!'
        assert stage <= self.stages, 'Exceeding the maximum stage number!'
        layers = []
        layers.append(Upsample())
        layers = conv_block(layers, in_features=self.nf(stage-1), out_features=self.nf(stage), kernel_size=3, stride=1, padding=1, pixel_norm=True)
        layers = conv_block(layers, in_features=self.nf(stage), out_features=self.nf(stage), kernel_size=3, stride=1, padding=1, pixel_norm=True)
        return  nn.Sequential(*layers)
    def grow_network(self):
        self.current_stage += 1
        assert self.current_stage <= self.stages, 'Exceeding the maximum stage number!'
        print('\ngrowing Generator...\n')
        # copy the trained layers except "to_rgb"
        new_model = deepcopy_exclude(self.model, ['to_rgb'])
        # old block (used for fade in)
        old_block = nn.Sequential()
        old_to_rgb = deepcopy_layers(self.model, ['to_rgb'])
        old_block.add_module('old_to_rgb', old_to_rgb[-1])
        old_block.add_module('old_upsample', Upsample())
        # new block to be faded in
        new_block = nn.Sequential()
        inter_block = self.intermediate_block(self.current_stage)
        new_block.add_module('new_block', inter_block)
        new_block.add_module('new_to_rgb', self.to_rgb_block(self.nf(self.current_stage)))
        # add fade in layer
        new_model.add_module('concat_block', ConcatTable(old_block, new_block))
        new_model.add_module('fadein', Fadein())
        del self.model
        self.model = new_model
    def flush_network(self):
        # once the fade in is finished, remove the old block and preserve the new block
        print('\nflushing Generator...\n')
        new_block = deepcopy_layers(self.model.concat_block.layer2, ['new_block'])
        new_to_rgb = deepcopy_layers(self.model.concat_block.layer2, ['new_to_rgb'])
        # copy the previous trained layers (before ConcatTable and Fadein)
        new_model = nn.Sequential()
        new_model = deepcopy_exclude(self.model, ['concat_block', 'fadein'])
        # preserve the new block
        layer_name = 'stage_{}'.format(self.current_stage)
        new_model.add_module(layer_name, new_block[-1])
        new_model.add_module('to_rgb', new_to_rgb[-1])
        del self.model
        self.model = new_model
    def forward(self, x, labels=None):
        assert len(x.size()) == 2 or len(x.size()) == 4, 'Invalid input size!'
        if len(x.size()) == 2:
            x = x.view(x.size(0), x.size(1), 1, 1)
        input = x
        if self.cond:
            assert labels is not None, 'Missing labels for conditional GAN!'
            cond = self.embedding(labels.long().view(-1)) # N --> N x C
            cond = cond.view(cond.size(0), cond.size(1), 1, 1).repeat(1, 1, x.size(2), x.size(3))
            cond = cond.float().detach()
            input = torch.cat((cond, x), dim=1)
        return self.model(input)
