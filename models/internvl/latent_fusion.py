import torch
import torch.nn as nn

class LatentFusionModule(nn.Module):
    def __init__(self, original_fusion_module, reasoning_module):
        super().__init__()
        self.original_fusion = original_fusion_module
        self.reasoning_module = reasoning_module

    def forward(self, vision_embeds, language_embeds):
        # Check for invalid (empty) language embeddings
        if language_embeds is None or language_embeds.shape[1] == 0:
            print("⚠️ Skipping batch: language_embeds is empty.")
            return vision_embeds  # or return None, or fused without reasoning

        fused = self.original_fusion(vision_embeds, language_embeds)

        # Optional: check fused shape
        if fused is None or fused.shape[1] == 0:
            print("⚠️ Skipping reasoning: fused embedding is empty.")
            return fused

        return self.reasoning_module(fused)
