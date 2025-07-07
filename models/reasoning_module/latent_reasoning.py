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

    def forward(self, vision_embeds, language_embeds):
        if language_embeds is None:
            print("⚠️ Skipping batch: language_embeds is None.")
            return vision_embeds

        if language_embeds.shape[0] == 0 or language_embeds.shape[1] == 0:
            print(f"⚠️ Skipping batch: language_embeds shape = {language_embeds.shape}")
            return vision_embeds

        fused = self.original_fusion(vision_embeds, language_embeds)

        if fused is None or fused.shape[0] == 0 or fused.shape[1] == 0:
            print(f"⚠️ Skipping reasoning: fused shape = {fused.shape if fused is not None else None}")
            return vision_embeds

        # Extra shape check to match vit and input embeddings before reasoning
        if fused.shape[1] != vision_embeds.shape[0]:
            print(f"⚠️ Shape mismatch: fused={fused.shape}, vision_embeds={vision_embeds.shape}")
            return vision_embeds

        return self.reasoning_module(fused)
