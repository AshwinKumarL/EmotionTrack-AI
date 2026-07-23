import pytest
import torch
import torch.nn as nn
from backend.emotion_engine.voice.model import (
    VoiceModelConfig,
    ConvBlock,
    EmotionCNN,
    ModelConfigurationError
)


def test_config_validation_success():
    """Verify that a valid configuration is instantiated correctly."""
    config = VoiceModelConfig(
        num_classes=8,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(32, 64, 128),
        kernel_sizes=(3, 3, 3),
        pool_sizes=(2, 2, 2),
        hidden_size=256
    )
    assert config.num_classes == 8
    assert config.input_channels == 1
    assert config.dropout_rate == 0.5
    assert config.filter_sizes == (32, 64, 128)
    assert config.hidden_size == 256


def test_config_validation_failures():
    """Verify that validation flags erroneous model configurations."""
    # Invalid num_classes
    with pytest.raises(ModelConfigurationError):
        VoiceModelConfig(
            num_classes=1,  # must be > 1
            input_channels=1,
            dropout_rate=0.5,
            filter_sizes=(32,),
            kernel_sizes=(3,),
            pool_sizes=(2,),
            hidden_size=128
        )
        
    # Invalid input_channels
    with pytest.raises(ModelConfigurationError):
        VoiceModelConfig(
            num_classes=8,
            input_channels=0,  # must be > 0
            dropout_rate=0.5,
            filter_sizes=(32,),
            kernel_sizes=(3,),
            pool_sizes=(2,),
            hidden_size=128
        )
        
    # Invalid dropout_rate (lower boundary)
    with pytest.raises(ModelConfigurationError):
        VoiceModelConfig(
            num_classes=8,
            input_channels=1,
            dropout_rate=-0.1,  # must be >= 0.0
            filter_sizes=(32,),
            kernel_sizes=(3,),
            pool_sizes=(2,),
            hidden_size=128
        )
        
    # Invalid dropout_rate (upper boundary)
    with pytest.raises(ModelConfigurationError):
        VoiceModelConfig(
            num_classes=8,
            input_channels=1,
            dropout_rate=1.0,  # must be < 1.0
            filter_sizes=(32,),
            kernel_sizes=(3,),
            pool_sizes=(2,),
            hidden_size=128
        )
        
    # Sequence length mismatch
    with pytest.raises(ModelConfigurationError):
        VoiceModelConfig(
            num_classes=8,
            input_channels=1,
            dropout_rate=0.5,
            filter_sizes=(32, 64),
            kernel_sizes=(3,),  # mismatch
            pool_sizes=(2, 2),
            hidden_size=128
        )


def test_dynamic_flatten_size():
    """Verify that modifying filters or pool configurations alters the flattened dimension automatically."""
    # Configuration A: 3 blocks, filter sizes (32, 64, 128)
    config_a = VoiceModelConfig(
        num_classes=8,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(32, 64, 128),
        kernel_sizes=(3, 3, 3),
        pool_sizes=(2, 2, 2),
        hidden_size=256
    )
    model_a = EmotionCNN(config_a)
    # Output of Block 3 before flatten: (B, 128, 16, 11) -> 128 * 16 * 11 = 22528
    assert model_a.flattened_dim == 22528

    # Configuration B: 2 blocks, filter sizes (16, 32)
    config_b = VoiceModelConfig(
        num_classes=8,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(16, 32),
        kernel_sizes=(3, 3),
        pool_sizes=(2, 2),
        hidden_size=256
    )
    model_b = EmotionCNN(config_b)
    # Output of Block 2 before flatten: (B, 32, 32, 23) -> 32 * 32 * 23 = 23552
    assert model_b.flattened_dim == 23552
    
    # Configuration C: Output collapse (Pooling sizes reduce spatial dimension below 1x1)
    config_c = VoiceModelConfig(
        num_classes=8,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(16, 32, 64, 128, 256, 512, 1024),  # Too many max-pools for 128x94 input
        kernel_sizes=(3, 3, 3, 3, 3, 3, 3),
        pool_sizes=(2, 2, 2, 2, 2, 2, 2),
        hidden_size=256
    )
    with pytest.raises(ModelConfigurationError):
        EmotionCNN(config_c)


def test_custom_input_dimensions():
    """Verify that non-default input_height and input_width are accepted and produce correct shapes."""
    config = VoiceModelConfig(
        num_classes=5,
        input_channels=1,
        dropout_rate=0.3,
        filter_sizes=(16, 32),
        kernel_sizes=(3, 3),
        pool_sizes=(2, 2),
        hidden_size=64,
        input_height=64,
        input_width=47
    )
    assert config.input_height == 64
    assert config.input_width == 47

    model = EmotionCNN(config)
    model.eval()

    x = torch.randn(2, 1, 64, 47)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 5)

    # Wrong spatial dims should raise ValueError
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 128, 94))


def test_forward_pass_and_output_shape():
    """Verify that the model processes standard input batches and yields correct shape."""
    config = VoiceModelConfig(
        num_classes=8,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(32, 64, 128),
        kernel_sizes=(3, 3, 3),
        pool_sizes=(2, 2, 2),
        hidden_size=256
    )
    model = EmotionCNN(config)
    
    # Batch size = 4
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 94)
    
    model.eval()  # Set to evaluation mode to disable dropout
    with torch.no_grad():
        logits = model(x)
        
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, 8)


def test_forward_invalid_input_dims():
    """Verify that forward pass raises ValueError on invalid input dimension sizes or layouts."""
    config = VoiceModelConfig(
        num_classes=8,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(32, 64, 128),
        kernel_sizes=(3, 3, 3),
        pool_sizes=(2, 2, 2),
        hidden_size=256
    )
    model = EmotionCNN(config)
    
    # 1. 3D input instead of 4D
    with pytest.raises(ValueError):
        model(torch.randn(1, 128, 94))
        
    # 2. Mismatched channels (e.g., 3 input channels instead of 1)
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 128, 94))
        
    # 3. Mismatched height or width
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 120, 94))
        
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 128, 100))


def test_gradient_propagation():
    """Verify that gradients can propagate backwards from logits to all layer weights."""
    config = VoiceModelConfig(
        num_classes=8,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(16, 32),
        kernel_sizes=(3, 3),
        pool_sizes=(2, 2),
        hidden_size=64
    )
    model = EmotionCNN(config)
    model.train()  # Make sure we are in training mode
    
    x = torch.randn(2, 1, 128, 94)
    logits = model(x)
    
    # Dummy target
    targets = torch.randint(0, 8, (2,))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    
    # Execute backward pass
    loss.backward()
    
    # Verify that trainable parameters have computed gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Output layers should have gradients computed
            assert param.grad is not None, f"Parameter {name} did not receive gradients."
            # Confirm gradients are non-zero
            assert torch.sum(torch.abs(param.grad)) > 0.0, f"Gradients for parameter {name} are zero."
