# Latent CoT Reasoning on InternVL

This project enhances the InternVL3-1B-Pretrained model with latent Chain-of-Thought reasoning inspired by Coconut.

## Structure
- `latent_fusion.py`: Injects latent reasoning into the fusion module.
- `latent_reasoning.py`: Applies iterative latent transformations.
- `train.py`: Trains on PuzzleWorld.

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python scripts/train.py
```

## Note
You do NOT need FlashAttention2 for basic runs on Windows.