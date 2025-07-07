# --- scripts/eval.py ---
import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from utils.dataloader import get_dataloaders
from accelerate import Accelerator

def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    batch_size = int(cfg["batch_size"])
    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(
        "OpenGVLab/InternVL3-1B-Pretrained",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        "OpenGVLab/InternVL3-1B-Pretrained",
        trust_remote_code=True
    )

    dataloaders = get_dataloaders(processor, batch_size=batch_size)
    model.eval()
    model, val_loader = accelerator.prepare(model, dataloaders["val"])

    correct = 0
    total = 0

    for batch in val_loader:
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=50
            )
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            targets = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)

            for pred, target in zip(preds, targets):
                if target.strip().lower() in pred.strip().lower():
                    correct += 1
                total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Validation Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
