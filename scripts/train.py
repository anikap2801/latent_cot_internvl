import sys
import os
import yaml
import traceback

# Add root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import torch
from transformers import AutoModel, AutoProcessor
from models.internvl.latent_fusion import LatentFusionModule
from models.reasoning_module.latent_reasoning import LatentReasoningModule
from utils.dataloader import get_dataloaders
from torch import optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter


def main():
    # Load config with absolute path
    config_path = os.path.join(ROOT_DIR, "configs", "config.yaml")
    with open(config_path, "r") as f:
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
    print("✅ Processor type:", type(processor))

    # Fallback to tokenizer if needed
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Freeze base encoders
    for name, param in model.named_parameters():
        if "vision_tower" in name or "language_model" in name:
            param.requires_grad = False

    # Replace fusion module with latent reasoning
    if hasattr(model, "fusion_module"):
        model.fusion_module = LatentFusionModule(
            model.fusion_module,
            LatentReasoningModule(dim=latent_dim, steps=latent_steps)
        )
    else:
        print("⚠️ Warning: model does not have attribute 'fusion_module'. Skipping fusion module replacement.")

    # Prepare dataloaders
    dataloaders = get_dataloaders(tokenizer, batch_size=batch_size)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, dataloaders['train']
    )

    # ✅ Use Kaggle-safe path
    writer = SummaryWriter(log_dir="/kaggle/working/runs/latent_cot")

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        print(f"\n🚀 Epoch {epoch+1} starting...")
        for step, batch in enumerate(train_loader):
            batch_size = batch["pixel_values"].size(0)
            image_flags = torch.ones(batch_size, 1, dtype=torch.bool).to(batch["pixel_values"].device)

            try:
                # Debug print shapes
                print(f"\n➡️ Step {step}")
                print("  input_ids:", batch["input_ids"].shape)
                print("  pixel_values:", batch["pixel_values"].shape)
                print("  attention_mask:", batch["attention_mask"].shape)
                print("  image_flags:", image_flags.shape)

                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    image_flags=image_flags,
                    labels=batch["input_ids"]
                )

                if outputs is None or not hasattr(outputs, "loss"):
                    print(f"⚠️ Skipping step {step}: model returned None or no loss.")
                    continue

            except Exception as e:
                print(f"❌ Error during forward pass at step {step}: {e}")
                traceback.print_exc()
                continue

            loss = outputs.loss
            epoch_loss += loss.item()

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)
        print(f"✅ Epoch {epoch + 1} completed - Avg Loss: {avg_loss:.4f}")

    writer.close()
    print("🏁 Training finished.")

if __name__ == "__main__":
    main()
