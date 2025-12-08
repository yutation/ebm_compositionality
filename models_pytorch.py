import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import utilities and building blocks from utils_pytorch
from utils_pytorch import (
    FLAGS, 
    swish, 
    spectral_norm_conv, 
    spectral_norm_fc,
    standard_transforms,
    apply_antialiasing,
    ConvBlock,
    ResBlock,
    AttentionBlock
)


class CubesNet(nn.Module):
    """Construct the convolutional network for energy-based models"""
    def __init__(self, num_filters=64, num_channels=3, label_size=6):
        super(CubesNet, self).__init__()
        
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.img_size = 64
        self.label_size = label_size
        print("label_size ", self.label_size)
        
        if not FLAGS.cclass:
            classes = 1
        else:
            classes = self.label_size
        
        # Initial convolution
        self.c1_pre = ConvBlock(num_channels, num_filters, kernel_size=3,
                                stride=1, padding=1, spec_norm=FLAGS.spec_norm,
                                classes=classes)
        
        # Residual blocks
        self.res_optim = ResBlock(num_filters, num_filters, adaptive=False,
                                 spec_norm=FLAGS.spec_norm, classes=classes)
        self.res_1 = ResBlock(num_filters, 2*num_filters, adaptive=True,
                             downsample=True, spec_norm=FLAGS.spec_norm, classes=classes)
        self.res_2 = ResBlock(2*num_filters, 2*num_filters, adaptive=False,
                             spec_norm=FLAGS.spec_norm, classes=classes)
        self.res_3 = ResBlock(2*num_filters, 2*num_filters, adaptive=False,
                             downsample=False, spec_norm=FLAGS.spec_norm, classes=classes)
        self.res_4 = ResBlock(2*num_filters, 4*num_filters, adaptive=True,
                             downsample=True, spec_norm=FLAGS.spec_norm, classes=classes)
        
        # Final FC layer
        fc = nn.Linear(4*num_filters, 1, bias=True)
        self.fc5 = fc  # No spectral norm for final layer
    
    def forward(self, inp, attention_mask=None, label=None, stop_grad=False):
        if FLAGS.swish_act:
            act = swish
        else:
            act = F.leaky_relu
        
        # Data augmentation if enabled
        if FLAGS.augment_vis:
            inp = standard_transforms(inp)
        
        # Antialias if enabled
        if FLAGS.antialias:
            inp = apply_antialiasing(inp, filter_size=3, stride=2)
        
        # Mask combination if enabled
        if FLAGS.comb_mask:
            attention_mask = F.softmax(attention_mask, dim=-1)
            # Reshape and apply mask
            B = inp.size(0)
            inp = inp.view(B, self.channels, self.img_size, self.img_size, 1)
            attention_mask = attention_mask.view(B, 1, self.img_size, self.img_size, FLAGS.cond_func)
            inp = (inp * attention_mask).permute(0, 4, 1, 2, 3).contiguous()
            inp = inp.view(B * FLAGS.cond_func, self.channels, self.img_size, self.img_size)
        
        # Forward pass
        if not FLAGS.cclass:
            label = None
        
        x = self.c1_pre(inp, label=label, activation=act)
        x = self.res_optim(x, label=label, act=act)
        x = self.res_1(x, label=label, act=act)
        x = self.res_2(x, label=label, act=act)
        x = self.res_3(x, label=label, act=act)
        x = self.res_4(x, label=label, act=act)
        
        x = act(x)
        x = torch.mean(x, dim=[2, 3])  # Global average pooling
        energy = self.fc5(x)
        
        # Combine energies if mask was used
        if FLAGS.comb_mask:
            energy = energy.view(-1, FLAGS.cond_func).sum(dim=1, keepdim=True)
        
        return energy


class CubesNetGen(nn.Module):
    """Generator network for image generation"""
    def __init__(self, num_filters=64, num_channels=3, label_size=6):
        super(CubesNetGen, self).__init__()
        
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.img_size = 64
        self.label_size = label_size
        print("label_size ", self.label_size)
        
        if not FLAGS.cclass:
            classes = 1
        else:
            classes = self.label_size
        
        # Initial FC layer
        self.fc_dense = spectral_norm_fc(nn.Linear(2*num_filters, 4*4*4*num_filters, bias=True))
        
        # Residual blocks with upsampling
        self.res_1 = ResBlock(4*num_filters, 2*num_filters, adaptive=True,
                             upsample=True, spec_norm=FLAGS.spec_norm, classes=classes)
        self.res_2 = ResBlock(2*num_filters, 2*num_filters, adaptive=False,
                             upsample=True, spec_norm=FLAGS.spec_norm, classes=classes)
        self.res_3 = ResBlock(2*num_filters, num_filters, adaptive=True,
                             upsample=True, spec_norm=FLAGS.spec_norm, classes=classes)
        self.res_4 = ResBlock(num_filters, num_filters, adaptive=False,
                             upsample=True, spec_norm=FLAGS.spec_norm, classes=classes)
        
        # Final convolution
        self.c4_out = ConvBlock(num_filters, num_channels, kernel_size=3,
                               stride=1, padding=1, spec_norm=FLAGS.spec_norm,
                               classes=classes)
    
    def forward(self, inp, label=None):
        if FLAGS.swish_act:
            act = swish
        else:
            act = F.leaky_relu
        
        if not FLAGS.cclass:
            label = None
        
        # FC layer and reshape
        x = self.fc_dense(inp)
        x = act(x)
        x = x.view(inp.size(0), 4*self.dim_hidden, 4, 4)
        
        # Residual blocks with upsampling
        x = self.res_1(x, label=label, act=act)
        x = self.res_2(x, label=label, act=act)
        x = self.res_3(x, label=label, act=act)
        x = self.res_4(x, label=label, act=act)
        
        # Final convolution (no activation)
        output = self.c4_out(x, label=label, activation=None)
        
        return output


class ResNet128(nn.Module):
    """ResNet architecture for 128x128 images"""
    def __init__(self, num_channels=3, num_filters=64, train=False, classes=1000):
        super(ResNet128, self).__init__()
        
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.dropout = train
        self.train_mode = train
        self.num_classes = classes
        
        print("set classes to be", classes)
        
        if not FLAGS.cclass:
            classes = 1
        else:
            classes = self.num_classes
        
        print("constructing weights with class number ", classes)
        
        # Initial convolution
        self.c1_pre = ConvBlock(num_channels, 64, kernel_size=3,
                               stride=1, padding=1, spec_norm=FLAGS.spec_norm,
                               classes=classes)
        
        # Residual blocks
        self.res_optim = ResBlock(64, num_filters, adaptive=False, downsample=True,
                                 spec_norm=FLAGS.spec_norm, classes=classes)
        
        # Optional attention
        if FLAGS.use_attention:
            self.atten = AttentionBlock(num_filters, num_filters // 2, 
                                       spec_norm=FLAGS.spec_norm)
        
        self.res_3 = ResBlock(num_filters, 2*num_filters, adaptive=True, downsample=True,
                             spec_norm=FLAGS.spec_norm, classes=classes)
        self.res_5 = ResBlock(2*num_filters, 4*num_filters, adaptive=True, downsample=True,
                             spec_norm=FLAGS.spec_norm, classes=classes)
        self.res_7 = ResBlock(4*num_filters, 8*num_filters, adaptive=True, downsample=True,
                             spec_norm=FLAGS.spec_norm, classes=classes)
        self.res_9 = ResBlock(8*num_filters, 8*num_filters, adaptive=False, downsample=True,
                             spec_norm=FLAGS.spec_norm, classes=classes)
        self.res_10 = ResBlock(8*num_filters, 8*num_filters, adaptive=False, downsample=False,
                              spec_norm=FLAGS.spec_norm, classes=classes)
        
        # Final FC layer
        self.fc5 = nn.Linear(8*num_filters, 1, bias=True)
    
    def forward(self, inp, label=None, latent=None):
        if FLAGS.swish_act:
            act = swish
        else:
            act = F.leaky_relu
        
        if not FLAGS.cclass:
            label = None
        
        dropout = self.dropout
        train = self.train_mode
        
        # Forward pass
        x = self.c1_pre(inp, label=label, activation=act)
        x = self.res_optim(x, label=label, act=act, dropout=dropout, train=train)
        
        if FLAGS.use_attention:
            x = self.atten(x)
        
        x = self.res_3(x, label=label, act=act, dropout=dropout, train=train)
        x = self.res_5(x, label=label, act=act, dropout=dropout, train=train)
        x = self.res_7(x, label=label, act=act, dropout=dropout, train=train)
        x = self.res_9(x, label=label, act=act, dropout=dropout, train=train)
        x = self.res_10(x, label=label, act=act, dropout=dropout, train=train)
        
        if FLAGS.swish_act:
            x = act(x)
        else:
            x = F.relu(x)
        
        x = torch.sum(x, dim=[2, 3])  # Global sum pooling
        energy = self.fc5(x)
        
        return energy


class CubesPredict(nn.Module):
    """Prediction network for object positions"""
    def __init__(self, num_channels=3, num_filters=64):
        super(CubesPredict, self).__init__()
        
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.datasource = FLAGS.datasource
        
        classes = 1
        
        # Convolutional layers
        self.c1_pre = ConvBlock(num_channels, 64, kernel_size=1, stride=1, padding=0,
                               spec_norm=False, classes=classes)
        self.c1 = ConvBlock(64, num_filters, kernel_size=4, stride=2, padding=1,
                           spec_norm=False, classes=classes)
        self.c2 = ConvBlock(num_filters, 2*num_filters, kernel_size=4, stride=2, padding=1,
                           spec_norm=False, classes=classes)
        self.c3 = ConvBlock(2*num_filters, 4*num_filters, kernel_size=4, stride=2, padding=1,
                           spec_norm=False, classes=classes)
        self.c4 = ConvBlock(4*num_filters, 4*num_filters, kernel_size=4, stride=2, padding=1,
                           spec_norm=False, classes=classes)
        
        # FC layers for position and logit prediction
        self.fc_dense_pos = nn.Linear(4*num_filters, 2*num_filters, bias=True)
        self.fc_dense_logit = nn.Linear(4*num_filters, 2*num_filters, bias=True)
        self.fc5_pos = nn.Linear(2*num_filters, 2, bias=True)
        self.fc5_logit = nn.Linear(2*num_filters, 1, bias=True)
    
    def forward(self, inp, label=None):
        if FLAGS.swish_act:
            act = swish
        else:
            act = F.leaky_relu
        
        # Reshape input
        x = inp.view(-1, self.channels, 64, 64)
        
        # Convolutional layers
        x = self.c1_pre(x, label=label, activation=act)
        x = self.c1(x, label=label, activation=act)
        x = self.c2(x, label=label, activation=act)
        x = self.c3(x, label=label, activation=act)
        x = self.c4(x, label=label, activation=act)
        
        # Global average pooling
        x = torch.mean(x, dim=[2, 3])
        
        # Position prediction
        h_pos = act(self.fc_dense_pos(x))
        pos = self.fc5_pos(h_pos)
        
        # Logit prediction
        h_logit = act(self.fc_dense_logit(x))
        logit = self.fc5_logit(h_logit)
        
        return logit, pos

