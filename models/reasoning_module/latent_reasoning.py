import torch
import torch.nn as nn

class LatentReasoningModule(nn.Module):
    def __init__(self, dim=768, steps=3, heads=8, use_feedback=False):
        super().__init__()
        self.steps = steps
        self.use_feedback = use_feedback

        # Multi-head self-attention and feed-forward layers
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Optional cross-modal feedback: a small GRU to fuse past + new thought
        if self.use_feedback:
            self.gru = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True)

    def forward(self, x, original=None, attention_mask=None):
        """
        x: [B, T, D] - latent fused representation
        original: optional [B, T, D] - original fused embedding (used for feedback)
        attention_mask: optional [B, T] - mask to restrict attention
        """
        for _ in range(self.steps):
            # Self-Attention with residual
            attn_output, _ = self.attn(x, x, x, key_padding_mask=~attention_mask if attention_mask is not None else None)
            x = x + attn_output  # residual connection

            # Feed-forward with residual
            x = x + self.ffn(x)

            # Optional: feedback from original features
            if self.use_feedback and original is not None:
                x, _ = self.gru(x, original.unsqueeze(0))  # feedback using GRU with original as initial hidden state

        return x
