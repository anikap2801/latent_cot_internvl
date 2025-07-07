import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from models.internvl.latent_fusion import LatentFusionModule
from models.reasoning_module.latent_reasoning import LatentReasoningModule
from utils.dataloader import get_dataloaders
from torch import optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("default")

def main():
    # Load config
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Extract config values
    latent_dim = int(cfg["latent_dim"])
    latent_steps = int(cfg["latent_steps"])
    batch_size = int(cfg["batch_size"])
    learning_rate = float(cfg["learning_rate"])
    epochs = int(cfg["epochs"])

    accelerator = Accelerator()

    # Load model
    model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL3-1B-Pretrained",
        trust_remote_code=True
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        "OpenGVLab/InternVL3-1B-Pretrained",
        trust_remote_code=True
    )
    print("Processor type:", type(processor))

    # Fallback to tokenizer if needed
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Freeze base encoders (text+image encoder)
    for name, param in model.named_parameters():
        if "vision_tower" in name or "language_model" in name:
            param.requires_grad = False

    # Replace fusion module
    if hasattr(model, "fusion_module"):
        model.fusion_module = LatentFusionModule(
            model.fusion_module,
            LatentReasoningModule(dim=latent_dim, steps=latent_steps)
        )

    # Prepare dataloaders
    dataloaders = get_dataloaders(tokenizer, batch_size=batch_size)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, dataloaders['train']
    )

    writer = SummaryWriter(log_dir="runs/latent_cot")

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            if step == 0:
                print("Batch info:")
                print("input_ids:", batch["input_ids"].shape)
                print("pixel_values:", batch["pixel_values"].shape)
                print("attention_mask:", batch["attention_mask"].shape)

            # Set image_flags to shape [batch_size, 1] with True values
            batch_size = batch["pixel_values"].size(0)
            image_flags = torch.ones(batch_size, 1, dtype=torch.bool).to(batch["pixel_values"].device)

            try:
                # Skip empty batch cases
                if batch["input_ids"].size(0) == 0 or batch["pixel_values"].size(0) == 0:
                    print(f"Step {step}: Empty batch found. Skipping.")
                    continue

                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    image_flags=image_flags,
                    labels=batch["input_ids"]
                )

                # Skip if model returned nothing
                if outputs is None or not hasattr(outputs, "loss"):
                    print(f"Skipping step {step} due to empty model output.")
                    continue

            except Exception as e:
                print("Model forward pass failed at step", step, "with error:", e)
                continue

            loss = outputs.loss
            epoch_loss += loss.item()

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    writer.close()

if __name__ == "__main__":
    main()
