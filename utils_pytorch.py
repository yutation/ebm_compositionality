"""PyTorch utility functions for EBM compositionality"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Simple FLAGS replacement for PyTorch (no TensorFlow dependency)
class FLAGS:
    spec_iter = 1  # Number of iterations to normalize spectrum of matrix
    spec_norm_val = 1.0  # Desired norm of matrices
    downsample = False  # Whether to do average pool downsampling
    spec_eval = False  # Set to true to prevent spectral updates
    spec_norm = True  # Use spectral normalization
    swish_act = False  # Use swish activation
    cclass = False  # Enable conditional class modeling
    use_attention = False  # Enable self-attention blocks
    augment_vis = False  # Enable visual augmentation
    antialias = False  # Enable antialiasing
    comb_mask = False  # Enable mask combination
    cond_func = 1  # Number of condition functions
    datasource = 'cubes'  # Dataset source


def set_seed(seed):
    """Set random seeds for reproducibility"""
    import numpy
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def swish(x):
    """Swish activation function"""
    return x * torch.sigmoid(x)


class ReplayBuffer(object):
    """Replay buffer for storing and sampling experiences"""
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, ims):
        """Add images to the replay buffer"""
        batch_size = ims.shape[0]
        if self._next_idx >= len(self._storage):
            self._storage.extend(list(ims))
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx +
                              batch_size] = list(ims)
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = list(ims)[:split_idx]
                self._storage[:batch_size - split_idx] = list(ims)[split_idx:]
        self._next_idx = (self._next_idx + ims.shape[0]) % self._maxsize

    def _encode_sample(self, idxes):
        """Encode a batch of samples from given indices"""
        ims = []
        for i in idxes:
            ims.append(self._storage[i])
        return np.array(ims)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)


def pixel_norm(x, epsilon=1e-8):
    """Pixel normalization"""
    return x * torch.rsqrt(torch.mean(x ** 2, dim=[1, 2], keepdim=True) + epsilon)


def group_norm(x, num_groups=32, eps=1e-6):
    """Group normalization for PyTorch tensors
    
    Args:
        x: Input tensor of shape [B, C, H, W]
        num_groups: Number of groups for normalization
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor
    """
    B, C, H, W = x.shape
    x = x.view(B, num_groups, C // num_groups, H, W)
    
    mean = x.mean(dim=[2, 3, 4], keepdim=True)
    var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
    
    x = (x - mean) / torch.sqrt(var + eps)
    x = x.view(B, C, H, W)
    
    return x


def layer_norm(x, eps=1e-6):
    """Layer normalization for PyTorch tensors
    
    Args:
        x: Input tensor of shape [B, C, H, W]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor
    """
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    var = x.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
    
    x = (x - mean) / torch.sqrt(var + eps)
    
    return x


def hw_flatten(x):
    """Flatten spatial dimensions
    
    Args:
        x: Input tensor of shape [B, C, H, W]
    
    Returns:
        Tensor of shape [B, H*W, C]
    """
    B, C, H, W = x.shape
    return x.view(B, C, H * W).permute(0, 2, 1)


def mse(pred, label):
    """Mean squared error loss"""
    return torch.mean((pred - label) ** 2)


# ============================================================================
# Neural Network Building Blocks (analogous to TensorFlow utils.py)
# ============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with optional normalization and conditional inputs
    
    This is the PyTorch equivalent of conv_block/smart_conv_block in TensorFlow utils.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=1, spec_norm=True, use_scale=True, classes=1):
        super(ConvBlock, self).__init__()
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                        stride=stride, padding=padding, bias=True)
        if spec_norm:
            conv = spectral_norm_conv(conv)
        self.conv = conv
        
        self.use_scale = use_scale
        self.classes = classes
        
        if classes != 1:
            self.scale = nn.Parameter(torch.ones(classes, out_channels))
            self.scale_bias = nn.Parameter(torch.zeros(classes, in_channels))
            self.extra_bias = nn.Parameter(torch.zeros(classes, in_channels))
        else:
            self.scale = nn.Parameter(torch.ones(out_channels))
            self.scale_bias = nn.Parameter(torch.zeros(in_channels))
            self.extra_bias = nn.Parameter(torch.zeros(in_channels))
            
        self.class_bias = nn.Parameter(torch.zeros(in_channels))
    
    def forward(self, x, label=None, use_extra_bias=False, activation=None):
        # Apply extra bias if needed
        if use_extra_bias and label is not None:
            if self.classes != 1:
                bias = torch.matmul(label, self.extra_bias)  # [B, in_channels]
                bias = bias.view(bias.size(0), bias.size(1), 1, 1)
            else:
                bias = self.extra_bias.view(1, -1, 1, 1)
            x = x + bias
        
        # Convolution
        x = self.conv(x)
        
        # Apply scale if needed
        if self.use_scale and label is not None:
            if self.classes != 1:
                scale = torch.matmul(label, self.scale) + self.class_bias
                scale = scale.view(scale.size(0), scale.size(1), 1, 1)
            else:
                scale = self.scale.view(1, -1, 1, 1)
            x = x * scale
        
        # Apply activation
        if activation is not None:
            x = activation(x)
        
        return x


class ResBlock(nn.Module):
    """Residual block with optional downsampling/upsampling
    
    This is the PyTorch equivalent of smart_res_block in TensorFlow utils.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 downsample=False, upsample=False, adaptive=True, 
                 spec_norm=True, classes=1):
        super(ResBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.upsample = upsample
        self.adaptive = adaptive
        
        # First conv block
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, 
                               stride=1, padding=kernel_size//2, 
                               spec_norm=spec_norm, classes=classes)
        
        # Second conv block (initialized with zeros in original)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 
                         stride=1, padding=kernel_size//2, bias=True)
        # Initialize with small values (approximating zero init)
        nn.init.normal_(conv2.weight, std=1e-10)
        if spec_norm:
            conv2 = spectral_norm_conv(conv2)
        self.conv2 = conv2
        
        # Scale and bias for conv2
        if classes != 1:
            self.scale2 = nn.Parameter(torch.ones(classes, out_channels))
            self.scale_bias2 = nn.Parameter(torch.zeros(classes, out_channels))
            self.extra_bias2 = nn.Parameter(torch.zeros(classes, out_channels))
        else:
            self.scale2 = nn.Parameter(torch.ones(out_channels))
            self.scale_bias2 = nn.Parameter(torch.zeros(out_channels))
            self.extra_bias2 = nn.Parameter(torch.zeros(out_channels))
        self.class_bias2 = nn.Parameter(torch.zeros(out_channels))
        self.classes = classes
        
        # Adaptive projection if channel dimensions change
        if adaptive and in_channels != out_channels:
            self.adaptive_conv = ConvBlock(in_channels, out_channels, kernel_size,
                                          stride=1, padding=kernel_size//2,
                                          spec_norm=spec_norm, classes=classes)
        else:
            self.adaptive_conv = None
    
    def forward(self, x, label=None, act=F.leaky_relu, dropout=False, train=False):
        # First convolution
        h = self.conv1(x, label=label, use_extra_bias=True, activation=None)
        h = act(h)
        
        if dropout and train:
            h = F.dropout(h, p=0.5, training=train)
        
        # Second convolution with extra bias
        if label is not None and self.classes != 1:
            bias = torch.matmul(label, self.extra_bias2)
            bias = bias.view(bias.size(0), bias.size(1), 1, 1)
            h = h + bias
        
        h = self.conv2(h)
        
        # Apply scale
        if label is not None and self.classes != 1:
            scale = torch.matmul(label, self.scale2) + self.class_bias2
            scale = scale.view(scale.size(0), scale.size(1), 1, 1)
            h = h * scale
        
        # Bypass connection
        if self.adaptive_conv is not None:
            residual = self.adaptive_conv(x, label=label, activation=None)
        else:
            residual = x
        
        # Add residual
        out = h + residual
        
        # Upsample or downsample
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        elif self.downsample:
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        
        # Final activation
        out = act(out)
        
        return out


class AttentionBlock(nn.Module):
    """Self-attention block
    
    This is the PyTorch equivalent of attention/smart_atten_block in TensorFlow utils.py
    """
    def __init__(self, in_channels, key_channels=None, spec_norm=True):
        super(AttentionBlock, self).__init__()
        
        if key_channels is None:
            key_channels = in_channels // 2
        
        # Query, Key, Value projections
        self.query = spectral_norm_conv(nn.Conv2d(in_channels, key_channels, 1))
        self.key = spectral_norm_conv(nn.Conv2d(in_channels, key_channels, 1))
        self.value = spectral_norm_conv(nn.Conv2d(in_channels, in_channels, 1))
        
        # Learnable gamma parameter
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Compute query, key, value
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, key_channels]
        k = self.key(x).view(B, -1, H * W)  # [B, key_channels, HW]
        v = self.value(x).view(B, -1, H * W)  # [B, C, HW]
        
        # Attention
        attn = torch.bmm(q, k)  # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(B, C, H, W)
        
        # Residual with learnable weight
        out = self.gamma * out + x
        
        return out


# Data transformations (PyTorch versions of TensorFlow transforms)

class Jitter:
    """Random jitter/crop transformation"""
    def __init__(self, d):
        self.d = d
    
    def __call__(self, x):
        """Apply random jitter to input tensor
        
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Jittered tensor
        """
        B, C, H, W = x.shape
        crop_h = H - self.d
        crop_w = W - self.d
        
        # Random crop
        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()
        
        return x[:, :, top:top+crop_h, left:left+crop_w]


class Pad:
    """Padding transformation"""
    def __init__(self, w, mode="reflect", constant_value=0.5):
        self.w = w
        self.mode = mode
        self.constant_value = constant_value
    
    def __call__(self, x):
        """Apply padding to input tensor
        
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Padded tensor
        """
        if self.mode == "reflect":
            return F.pad(x, (self.w, self.w, self.w, self.w), mode='reflect')
        elif self.mode == "constant":
            if self.constant_value == "uniform":
                value = torch.rand(1).item()
            else:
                value = self.constant_value
            return F.pad(x, (self.w, self.w, self.w, self.w), mode='constant', value=value)
        else:
            return F.pad(x, (self.w, self.w, self.w, self.w), mode=self.mode)


class RandomScale:
    """Random scaling transformation"""
    def __init__(self, scales):
        self.scales = scales
    
    def __call__(self, x):
        """Apply random scaling to input tensor
        
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Scaled tensor
        """
        scale = random.choice(self.scales)
        B, C, H, W = x.shape
        new_h = int(H * scale)
        new_w = int(W * scale)
        
        return F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)


class RandomRotate:
    """Random rotation transformation"""
    def __init__(self, angles, units="degrees"):
        self.angles = angles
        self.units = units
    
    def __call__(self, x):
        """Apply random rotation to input tensor
        
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Rotated tensor
        """
        angle = random.choice(self.angles)
        
        if self.units.lower() == "degrees":
            angle_rad = np.pi * angle / 180.0
        else:
            angle_rad = angle
        
        # Create rotation matrix
        theta = torch.tensor([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0]
        ], dtype=x.dtype, device=x.device)
        
        theta = theta.unsqueeze(0).repeat(x.size(0), 1, 1)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        rotated = F.grid_sample(x, grid, align_corners=False)
        
        return rotated


class Compose:
    """Compose multiple transformations"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        """Apply all transformations in sequence
        
        Args:
            x: Input tensor
        
        Returns:
            Transformed tensor
        """
        for transform in self.transforms:
            x = transform(x)
        return x


# Standard transforms for data augmentation
standard_transforms = Compose([
    Pad(6, mode="constant", constant_value=0.5),
    Jitter(4),
    RandomRotate(list(range(-10, 11)) + 5 * [0]),
    Jitter(2),
])


# Antialiasing filters

def get_stride_3_filter():
    """Get 3x3 antialiasing filter"""
    stride_3 = np.array([1, 2, 1], dtype=np.float32)
    stride_3 = stride_3[:, None] * stride_3[None, :]
    stride_3 = stride_3 / stride_3.sum()
    stride_3 = stride_3[None, None, :, :]  # [1, 1, 3, 3]
    return torch.from_numpy(stride_3)


def get_stride_5_filter():
    """Get 5x5 antialiasing filter"""
    stride_5 = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    stride_5 = stride_5[:, None] * stride_5[None, :]
    stride_5 = stride_5 / stride_5.sum()
    stride_5 = stride_5[None, None, :, :]  # [1, 1, 5, 5]
    return torch.from_numpy(stride_5)


def apply_antialiasing(x, filter_size=3, stride=2):
    """Apply antialiasing filter to input
    
    Args:
        x: Input tensor of shape [B, C, H, W]
        filter_size: Size of antialiasing filter (3 or 5)
        stride: Stride for downsampling
    
    Returns:
        Filtered and downsampled tensor
    """
    if filter_size == 3:
        aa_filter = get_stride_3_filter()
    else:
        aa_filter = get_stride_5_filter()
    
    aa_filter = aa_filter.to(x.device).to(x.dtype)
    C = x.size(1)
    aa_filter = aa_filter.repeat(C, 1, 1, 1)
    
    padding = filter_size // 2
    x = F.conv2d(x, aa_filter, stride=stride, padding=padding, groups=C)
    
    return x


# Helper functions for loading images

def get_images(paths, labels, nb_samples=None, shuffle=True):
    """Get list of image paths with labels
    
    Args:
        paths: List of directory paths
        labels: List of labels corresponding to paths
        nb_samples: Optional number of samples per directory
        shuffle: Whether to shuffle the results
    
    Returns:
        List of (label, image_path) tuples
    """
    import os
    
    if nb_samples is not None:
        def sampler(x): return random.sample(x, nb_samples)
    else:
        def sampler(x): return x
    
    images = [(i, os.path.join(path, image))
              for i, path in zip(labels, paths)
              for image in sampler(os.listdir(path))]
    
    if shuffle:
        random.shuffle(images)
    
    return images


# Spectral normalization utilities (PyTorch has built-in support)

def spectral_norm_conv(module, name='weight', n_power_iterations=1):
    """Apply spectral normalization to convolutional layer
    
    Args:
        module: nn.Conv2d module
        name: Name of weight parameter
        n_power_iterations: Number of power iterations
    
    Returns:
        Module with spectral normalization applied
    """
    if FLAGS.spec_norm:
        return nn.utils.spectral_norm(module, name=name, n_power_iterations=n_power_iterations)
    return module


def spectral_norm_fc(module, name='weight', n_power_iterations=2):
    """Apply spectral normalization to fully connected layer
    
    Args:
        module: nn.Linear module
        name: Name of weight parameter
        n_power_iterations: Number of power iterations (higher for FC layers)
    
    Returns:
        Module with spectral normalization applied
    """
    if FLAGS.spec_norm:
        return nn.utils.spectral_norm(module, name=name, n_power_iterations=n_power_iterations)
    return module


# Gradient utilities

def average_gradients(grads_list):
    """Average gradients across multiple computations
    
    Args:
        grads_list: List of gradient dictionaries
    
    Returns:
        Dictionary of averaged gradients
    """
    if not grads_list:
        return {}
    
    avg_grads = {}
    for key in grads_list[0].keys():
        grad_stack = torch.stack([g[key] for g in grads_list if g[key] is not None])
        avg_grads[key] = torch.mean(grad_stack, dim=0)
    
    return avg_grads


# Model checkpoint utilities

def save_checkpoint(model, optimizer, epoch, path):
    """Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        path: Path to checkpoint file
    
    Returns:
        Starting epoch number
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path}, starting from epoch {epoch}")
    return epoch


def optimistic_restore(model, checkpoint_path, strict=False):
    """Restore model weights optimistically (only matching keys)
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        strict: Whether to strictly match all keys
    
    Returns:
        Number of loaded parameters
    """
    checkpoint = torch.load(checkpoint_path)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model_dict = model.state_dict()
    
    # Filter out keys that don't match
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            pretrained_dict[k] = v
        else:
            print(f"Skipping key {k}: shape mismatch or not found")
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=strict)
    
    print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} parameters")
    return len(pretrained_dict)


# Normalization functions

class GroupNorm2d(nn.Module):
    """Group Normalization layer"""
    def __init__(self, num_channels, num_groups=32, eps=1e-6):
        super(GroupNorm2d, self).__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.num_groups, C // self.num_groups, H, W)
        
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(B, C, H, W)
        
        # Apply learned affine transform
        x = x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        
        return x


class LayerNorm2d(nn.Module):
    """Layer Normalization for 2D inputs"""
    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        var = x.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learned affine transform
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return x

