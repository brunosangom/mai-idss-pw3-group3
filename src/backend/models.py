import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class BaseModel(nn.Module):
    """Base class for all wildfire prediction models."""
    
    def __init__(self, num_features: int, window_size: int, device: torch.device):
        super().__init__()
        self.num_features = num_features
        self.window_size = window_size
        self.device = device
    
    def forward(self, sequences: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            sequences: Input sequences of shape (batch, seq_len, num_features)
            targets: Target sequences for autoregressive models (batch, seq_len)
            
        Returns:
            Predictions of shape (batch, seq_len)
        """
        raise NotImplementedError("Subclasses must implement forward method")


class TransformerModel(BaseModel):
    """Transformer-based model for wildfire prediction.
    
    Note: The `targets` parameter in forward() is currently unused. 
    It was designed for potential autoregressive decoding but the current
    implementation uses an encoder-only architecture.
    """
    
    def __init__(self, num_features: int, window_size: int, device: torch.device, 
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__(num_features, window_size, device)
        
        self.d_model = d_model
        
        # Source embedding layer to transform from num_features to d_model
        self.source_embedder = nn.Linear(num_features, d_model)
        
        # Positional embedding to add positional information to the input sequences
        self.pos_emb = nn.Parameter(torch.randn(1, window_size, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer for binary classification
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequences: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the Transformer model.
        
        Args:
            sequences: Input features (batch, seq_len, num_features)
            targets: Unused, kept for API compatibility
        """
        # Apply source embedding to transform from num_features to d_model
        x = self.source_embedder(sequences)  # (batch, seq_len, d_model)
        
        # Add positional embeddings
        x = x + self.pos_emb  # (batch, seq_len, d_model)
        
        # Pass through transformer
        transformer_out = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Apply output layer
        outputs = self.output_layer(transformer_out)  # (batch, seq_len, 1)
        
        # Reshape outputs for loss calculation
        outputs = outputs.squeeze(-1)  # (batch, seq_len)
        return outputs


class LSTMModel(BaseModel):
    """LSTM-based model for wildfire prediction."""
    
    def __init__(self, num_features: int, window_size: int, device: torch.device,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__(num_features, window_size, device)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer for binary classification
        lstm_output_size = hidden_size * self.num_directions
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequences: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the LSTM model.
        
        Args:
            sequences: Input features (batch, seq_len, num_features)
            targets: Unused, kept for API compatibility
        """
        # Pass through LSTM
        # lstm_out shape: (batch, seq_len, hidden_size * num_directions)
        lstm_out, _ = self.lstm(sequences)
        
        # Apply output layer to each time step
        outputs = self.output_layer(lstm_out)  # (batch, seq_len, 1)
        
        # Reshape outputs for loss calculation
        outputs = outputs.squeeze(-1)  # (batch, seq_len)
        return outputs


def create_model(model_config: Dict[str, Any], num_features: int, 
                 window_size: int, device: torch.device) -> BaseModel:
    """
    Factory function to create a model based on configuration.
    
    Args:
        model_config: Model configuration dictionary containing 'name' and 'params'
        num_features: Number of input features
        window_size: Length of input sequences
        device: Device to place the model on
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model name is not supported
    """
    model_name = model_config['name']
    params = model_config.get('params', {})
    
    if model_name == 'Transformer':
        model = TransformerModel(
            num_features=num_features,
            window_size=window_size,
            device=device,
            d_model=params.get('d_model', 64),
            nhead=params.get('nhead', 4),
            num_layers=params.get('num_layers', 2),
            dim_feedforward=params.get('dim_feedforward', 256),
            dropout=params.get('dropout', 0.1)
        )
    elif model_name == 'LSTM':
        model = LSTMModel(
            num_features=num_features,
            window_size=window_size,
            device=device,
            hidden_size=params.get('hidden_size', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.1),
            bidirectional=params.get('bidirectional', False)
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported. "
                        f"Available models: Transformer, LSTM")
    
    return model.to(device)
