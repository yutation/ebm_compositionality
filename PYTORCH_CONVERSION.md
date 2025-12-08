# TensorFlow to PyTorch Conversion Guide

This document describes the conversion of the EBM compositionality models from TensorFlow to PyTorch.

## Files Created

1. **models_pytorch.py** - PyTorch version of all model architectures (imports from utils_pytorch.py)
2. **utils_pytorch.py** - PyTorch utility functions and helper classes (used by models_pytorch.py)
3. **example_pytorch.py** - Example usage and test suite for all models
4. **requirements_pytorch.txt** - Python package dependencies for PyTorch version

## Module Dependencies

```
models_pytorch.py
  └── imports from utils_pytorch.py:
      ├── FLAGS (configuration)
      ├── swish (activation function)
      ├── spectral_norm_conv (weight normalization)
      ├── spectral_norm_fc (weight normalization)
      ├── standard_transforms (data augmentation)
      └── apply_antialiasing (downsampling filter)
```

## Models Converted

### 1. CubesNet
Energy-based model for 64x64 images with the following features:
- Residual blocks with optional downsampling
- Conditional class support via label parameter
- Optional attention masks for compositional reasoning
- Spectral normalization support

**Usage:**
```python
from models_pytorch import CubesNet
import torch

model = CubesNet(num_filters=64, num_channels=3, label_size=6)
model = model.cuda()

# Forward pass
images = torch.randn(32, 3, 64, 64).cuda()
labels = torch.randn(32, 6).cuda()  # if FLAGS.cclass is True
energy = model(images, label=labels)
```

### 2. CubesNetGen
Generator network for image synthesis:
- Transposed convolutions for upsampling
- Residual blocks with learnable skip connections
- Generates 64x64 images from latent codes

**Usage:**
```python
from models_pytorch import CubesNetGen
import torch

model = CubesNetGen(num_filters=64, num_channels=3, label_size=6)
model = model.cuda()

# Forward pass
latent = torch.randn(32, 128).cuda()
labels = torch.randn(32, 6).cuda()  # if FLAGS.cclass is True
generated_images = model(latent, label=labels)
```

### 3. ResNet128
ResNet architecture for 128x128 images:
- Deeper architecture with 8x base filters in final layers
- Optional self-attention block
- Configurable dropout for training
- Designed for ImageNet-scale classification/energy modeling

**Usage:**
```python
from models_pytorch import ResNet128
import torch

model = ResNet128(num_channels=3, num_filters=64, train=True, classes=1000)
model = model.cuda()

# Forward pass
images = torch.randn(32, 3, 128, 128).cuda()
energy = model(images)
```

### 4. CubesPredict
Prediction network for object localization:
- Predicts both position (x, y) and existence logit
- Strided convolutions for efficient downsampling
- Dual-head architecture for multi-task learning

**Usage:**
```python
from models_pytorch import CubesPredict
import torch

model = CubesPredict(num_channels=3, num_filters=64)
model = model.cuda()

# Forward pass
images = torch.randn(32, 3, 64, 64).cuda()
logit, position = model(images)
```

## Key Conversion Notes

### Architecture Changes

1. **Module Structure:**
   - TensorFlow's weight dictionaries → PyTorch nn.Module with parameters
   - `construct_weights()` → moved to `__init__()`
   - Manual weight management → automatic via PyTorch

2. **Convolutions:**
   - `tf.nn.conv2d` → `nn.Conv2d`
   - TensorFlow NHWC format → PyTorch NCHW format
   - Stride format: `[1, 2, 2, 1]` → `stride=2`

3. **Normalization:**
   - TensorFlow spectral norm → `nn.utils.spectral_norm`
   - Custom group norm → PyTorch implementation
   - Batch norm → `nn.BatchNorm2d`

4. **Activations:**
   - `tf.nn.leaky_relu` → `F.leaky_relu`
   - Custom swish → PyTorch implementation

5. **Pooling:**
   - `tf.layers.average_pooling2d` → `F.avg_pool2d`
   - `tf.reduce_mean` → `torch.mean`

6. **Upsampling:**
   - `tf.image.resize_nearest_neighbor` → `F.interpolate`

### Utility Functions

The `utils_pytorch.py` file provides:

- **ReplayBuffer**: Experience replay for training
- **Data Transforms**: Jitter, padding, rotation, scaling
- **Normalization**: Group norm, layer norm, pixel norm
- **Antialiasing**: Blur filters for downsampling
- **Checkpoint Utils**: Save/load model states
- **Spectral Norm**: Wrapper functions for weight normalization

### FLAGS Compatibility

The PyTorch version maintains compatibility with TensorFlow FLAGS:
- `FLAGS.spec_norm` - Enable spectral normalization
- `FLAGS.swish_act` - Use swish activation
- `FLAGS.cclass` - Enable conditional class modeling
- `FLAGS.use_attention` - Enable self-attention blocks
- `FLAGS.augment_vis` - Enable visual augmentation
- `FLAGS.antialias` - Enable antialiasing
- `FLAGS.comb_mask` - Enable mask combination

## Migration Guide

### Converting Training Code

**TensorFlow:**
```python
# TensorFlow 1.x style
import tensorflow as tf
from models import CubesNet

model = CubesNet()
weights = model.construct_weights()
energy = model.forward(images, weights, attention_mask=None, label=labels)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
train_op = optimizer.minimize(loss)
```

**PyTorch:**
```python
# PyTorch style
import torch
import torch.optim as optim
from models_pytorch import CubesNet

model = CubesNet().cuda()
energy = model(images, label=labels)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss.backward()
optimizer.step()
```

### Key Differences

1. **Automatic Differentiation:**
   - TensorFlow: Builds computation graph, requires session
   - PyTorch: Dynamic computation graph, immediate execution

2. **Training Loop:**
   - TensorFlow: `sess.run([train_op, loss])`
   - PyTorch: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`

3. **Device Management:**
   - TensorFlow: Automatic GPU placement
   - PyTorch: Explicit `.cuda()` or `.to(device)`

4. **Data Format:**
   - TensorFlow: NHWC (batch, height, width, channels)
   - PyTorch: NCHW (batch, channels, height, width)

## Testing Conversion

To verify the conversion is correct:

```python
import torch
from models_pytorch import CubesNet

# Create model
model = CubesNet(num_filters=64, num_channels=3, label_size=6)

# Test forward pass
batch_size = 8
images = torch.randn(batch_size, 3, 64, 64)
labels = torch.randn(batch_size, 6)

# Set FLAGS as needed
import tensorflow as tf
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
FLAGS.cclass = True
FLAGS.spec_norm = True

# Forward pass
energy = model(images, label=labels)
print(f"Energy shape: {energy.shape}")  # Should be [8, 1]

# Test backward pass
energy.sum().backward()
print("Backward pass successful!")
```

## Known Limitations

1. **FLAGS System**: Still uses TensorFlow's FLAGS. Consider migrating to argparse or hydra for pure PyTorch.

2. **Data Transforms**: The `standard_transforms` in utils_pytorch.py are implemented but may need tuning to exactly match TensorFlow behavior.

3. **Spectral Normalization**: PyTorch's built-in spectral norm uses power iteration. The behavior should be similar but may not be identical to the TensorFlow implementation.

4. **Weight Initialization**: PyTorch default initialization differs from TensorFlow. For exact reproduction, you may need to initialize from TensorFlow checkpoints.

## Converting Checkpoints

To convert TensorFlow checkpoints to PyTorch:

```python
import tensorflow as tf
import torch
from models_pytorch import CubesNet

# Load TensorFlow checkpoint
reader = tf.train.NewCheckpointReader('path/to/tf/checkpoint')
var_to_shape_map = reader.get_variable_to_shape_map()

# Create PyTorch model
model = CubesNet()

# Manual weight mapping (requires understanding of weight names)
# This is dataset/architecture specific
state_dict = {}
for key in var_to_shape_map:
    # Map TensorFlow variable names to PyTorch parameter names
    pytorch_key = map_tf_to_pytorch_key(key)
    state_dict[pytorch_key] = torch.from_numpy(reader.get_tensor(key))

model.load_state_dict(state_dict, strict=False)
```

## Future Improvements

1. Replace FLAGS with argparse or OmegaConf
2. Add torchvision compatibility for data loading
3. Implement distributed training support
4. Add mixed precision training (AMP)
5. Create comprehensive test suite comparing TF and PyTorch outputs
6. Optimize memory usage with gradient checkpointing

## References

- Original TensorFlow implementation: `models.py`, `utils.py`
- PyTorch documentation: https://pytorch.org/docs/
- Spectral normalization: https://arxiv.org/abs/1802.05957

