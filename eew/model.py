"""
Transformer Transformer Model Implementation

Following the architecture from Wu et al., 2025:
- Input: 3-channel waveform (X, Y, Z) × n timesteps
- Embedding: Linear projection to d_model=64
- Positional encoding: Sinusoidal
- Transformer encoder: Multi-head attention (h=8), LayerNorm, FFN
- Classifier: 2 FC layers with MC Dropout
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in Vaswani et al. (2017).
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with pre-norm architecture.
    
    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> Dropout -> + (residual) ->
        -> LayerNorm -> FeedForward -> Dropout -> + (residual) -> output
    """
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Pre-norm + Multi-head attention + residual
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm + Feed-forward + residual
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        
        return x


class Transformer(nn.Module):
    """
    Transformer: Transformer-based Earthquake Detection Model
    
    Architecture following Wu et al., 2025:
    - Input: (batch, 3, seq_len) - 3 channels (X, Y, Z)
    - Embedding: Linear projection from 3 to d_model=64
    - Positional encoding added
    - Transformer encoder with multiple layers
    - Classifier: 2 FC layers (64 -> 200 -> 200 -> 1)
    - MC Dropout for uncertainty estimation
    
    Target: ~235k parameters
    """
    
    def __init__(
        self,
        seq_len=200,
        input_channels=3,
        d_model=64,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        classifier_hidden=200,
        mc_dropout=True
    ):
        """
        Args:
            seq_len: Sequence length (default 200 for 2s @ 100Hz)
            input_channels: Number of input channels (default 3: X, Y, Z)
            d_model: Model dimension (default 64)
            nhead: Number of attention heads (default 8)
            num_encoder_layers: Number of transformer encoder layers (default 4)
            dim_feedforward: FFN hidden dimension (default 256 = 4 * d_model)
            dropout: Dropout rate (default 0.1)
            classifier_hidden: Hidden size of classifier layers (default 200)
            mc_dropout: Enable MC Dropout for uncertainty estimation
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.d_model = d_model
        self.mc_dropout = mc_dropout
        
        # Input embedding: Map from input_channels to d_model at each timestep
        # Input: (batch, 3, seq_len) -> (batch, seq_len, 3) -> (batch, seq_len, d_model)
        self.embedding = nn.Linear(input_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Global pooling (take the mean across sequence dimension)
        # Alternative: use first token or learnable [CLS] token
        self.pooling = 'mean'  # 'mean', 'max', or 'first'
        
        # Classifier (decoder): 2 FC layers with dropout
        # d_model -> classifier_hidden -> classifier_hidden -> 1
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 3, seq_len) or (batch_size, seq_len, 3)
            return_attention: If True, return attention weights (not implemented)
        
        Returns:
            logits: Output logits (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Ensure input is (batch, seq_len, channels)
        if x.size(1) == self.input_channels:
            x = x.transpose(1, 2)  # (batch, 3, seq_len) -> (batch, seq_len, 3)
        
        # Embedding: (batch, seq_len, 3) -> (batch, seq_len, d_model)
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Pooling: (batch, seq_len, d_model) -> (batch, d_model)
        if self.pooling == 'mean':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]
        elif self.pooling == 'first':
            x = x[:, 0, :]
        
        # Classifier: (batch, d_model) -> (batch, 1)
        logits = self.classifier(x)
        
        return logits
    
    def predict_with_uncertainty(self, x, n_passes=10):
        """
        Predict with MC Dropout uncertainty estimation.
        
        Performs multiple forward passes with dropout enabled to estimate
        predictive uncertainty.
        
        Args:
            x: Input tensor (batch_size, 3, seq_len)
            n_passes: Number of MC dropout passes (default 10)
        
        Returns:
            mean_prob: Mean predicted probability (batch_size,)
            std_prob: Standard deviation of predictions (batch_size,)
            entropy: Predictive entropy (batch_size,)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = self.forward(x)
                probs = torch.sigmoid(logits).squeeze()
                predictions.append(probs)
        
        # Stack predictions: (n_passes, batch_size)
        predictions = torch.stack(predictions)
        
        # Compute statistics
        mean_prob = predictions.mean(dim=0)
        std_prob = predictions.std(dim=0)
        
        # Compute predictive entropy
        # H = -p*log(p) - (1-p)*log(1-p)
        eps = 1e-10
        entropy = -(mean_prob * torch.log(mean_prob + eps) + 
                   (1 - mean_prob) * torch.log(1 - mean_prob + eps))
        
        return mean_prob, std_prob, entropy


def create_Transformer_model(
    seq_len=200,
    input_channels=3,
    d_model=64,
    nhead=8,
    num_encoder_layers=4,
    dim_feedforward=256,
    dropout=0.1,
    classifier_hidden=200,
    mc_dropout=True
):
    """
    Factory function to create Transformer model.
    
    Default configuration aims for ~235k parameters as in the paper.
    
    Returns:
        Transformer model
    """
    model = Transformer(
        seq_len=seq_len,
        input_channels=input_channels,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        classifier_hidden=classifier_hidden,
        mc_dropout=mc_dropout
    )
    
    return model


if __name__ == '__main__':
    # Test model
    print("Testing Transformer model...")
    
    model = create_Transformer_model()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 200)
    
    print(f"Input shape: {x.shape}")
    
    # Standard forward
    logits = model(x)
    print(f"Output logits shape: {logits.shape}")
    print(f"Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # MC Dropout prediction
    mean_prob, std_prob, entropy = model.predict_with_uncertainty(x, n_passes=10)
    print(f"Mean probability shape: {mean_prob.shape}")
    print(f"Mean uncertainty (entropy): {entropy.mean():.3f}")
    
    print("\nTest passed!")
