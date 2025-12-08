"""
Example script demonstrating PyTorch model usage
"""
import torch
import torch.nn as nn
import torch.optim as optim
from models_pytorch import CubesNet, CubesNetGen, ResNet128, CubesPredict
from utils_pytorch import FLAGS


def test_cubes_net():
    """Test CubesNet energy model"""
    print("=" * 50)
    print("Testing CubesNet")
    print("=" * 50)
    
    # Configure FLAGS
    FLAGS.cclass = False
    FLAGS.spec_norm = True
    FLAGS.swish_act = False
    
    # Create model
    model = CubesNet(num_filters=64, num_channels=3, label_size=6)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model on device: {device}")
    
    # Create dummy input
    batch_size = 8
    images = torch.randn(batch_size, 3, 64, 64).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        energy = model(images)
    
    print(f"Input shape: {images.shape}")
    print(f"Energy shape: {energy.shape}")
    print(f"Energy values: {energy.mean().item():.4f} ± {energy.std().item():.4f}")
    
    # Test backward pass
    model.train()
    energy = model(images)
    loss = energy.mean()
    loss.backward()
    print(f"Backward pass successful!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()


def test_cubes_net_gen():
    """Test CubesNetGen generator"""
    print("=" * 50)
    print("Testing CubesNetGen")
    print("=" * 50)
    
    # Configure FLAGS
    FLAGS.cclass = False
    FLAGS.spec_norm = True
    
    # Create model
    model = CubesNetGen(num_filters=64, num_channels=3, label_size=6)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy latent code
    batch_size = 8
    latent = torch.randn(batch_size, 128).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        generated = model(latent)
    
    print(f"Latent shape: {latent.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Generated range: [{generated.min().item():.4f}, {generated.max().item():.4f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()


def test_resnet128():
    """Test ResNet128 model"""
    print("=" * 50)
    print("Testing ResNet128")
    print("=" * 50)
    
    # Configure FLAGS
    FLAGS.cclass = False
    FLAGS.spec_norm = True
    FLAGS.use_attention = False
    
    # Create model
    model = ResNet128(num_channels=3, num_filters=64, train=False, classes=1000)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input
    batch_size = 4  # Smaller batch for 128x128
    images = torch.randn(batch_size, 3, 128, 128).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        energy = model(images)
    
    print(f"Input shape: {images.shape}")
    print(f"Energy shape: {energy.shape}")
    print(f"Energy values: {energy.mean().item():.4f} ± {energy.std().item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()


def test_cubes_predict():
    """Test CubesPredict model"""
    print("=" * 50)
    print("Testing CubesPredict")
    print("=" * 50)
    
    # Configure FLAGS
    FLAGS.datasource = 'cubes'
    
    # Create model
    model = CubesPredict(num_channels=3, num_filters=64)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input
    batch_size = 8
    images = torch.randn(batch_size, 3, 64, 64).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logit, position = model(images)
    
    print(f"Input shape: {images.shape}")
    print(f"Logit shape: {logit.shape}")
    print(f"Position shape: {position.shape}")
    print(f"Logit values: {logit.mean().item():.4f} ± {logit.std().item():.4f}")
    print(f"Position values: {position.mean().item():.4f} ± {position.std().item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()


def test_conditional_model():
    """Test CubesNet with conditional labels"""
    print("=" * 50)
    print("Testing CubesNet with Conditional Labels")
    print("=" * 50)
    
    # Configure FLAGS for conditional modeling
    FLAGS.cclass = True
    FLAGS.spec_norm = True
    
    # Create model
    model = CubesNet(num_filters=64, num_channels=3, label_size=6)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input with labels
    batch_size = 8
    images = torch.randn(batch_size, 3, 64, 64).to(device)
    # One-hot encoded labels
    labels = torch.zeros(batch_size, 6).to(device)
    labels[torch.arange(batch_size), torch.randint(0, 6, (batch_size,))] = 1.0
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        energy = model(images, label=labels)
    
    print(f"Input shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Energy shape: {energy.shape}")
    print(f"Energy values: {energy.mean().item():.4f} ± {energy.std().item():.4f}")
    print()


def test_training_loop():
    """Demonstrate a simple training loop"""
    print("=" * 50)
    print("Testing Training Loop")
    print("=" * 50)
    
    # Configure FLAGS
    FLAGS.cclass = False
    FLAGS.spec_norm = True
    
    # Create model
    model = CubesNet(num_filters=32, num_channels=3, label_size=6)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Dummy training loop
    batch_size = 4
    num_steps = 5
    
    print(f"Running {num_steps} training steps...")
    for step in range(num_steps):
        # Generate dummy data
        positive_images = torch.randn(batch_size, 3, 64, 64).to(device)
        negative_images = torch.randn(batch_size, 3, 64, 64).to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute energies
        positive_energy = model(positive_images)
        negative_energy = model(negative_images)
        
        # Contrastive loss (want positive energy low, negative energy high)
        loss = positive_energy.mean() - negative_energy.mean()
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        print(f"Step {step+1}: Loss = {loss.item():.4f}, "
              f"E+ = {positive_energy.mean().item():.4f}, "
              f"E- = {negative_energy.mean().item():.4f}")
    
    print("Training loop completed!")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("PyTorch Models Test Suite")
    print("=" * 50 + "\n")
    
    # Run all tests
    test_cubes_net()
    test_cubes_net_gen()
    test_resnet128()
    test_cubes_predict()
    test_conditional_model()
    test_training_loop()
    
    print("=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)

