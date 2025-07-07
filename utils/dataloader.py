import os, json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PuzzleWorldDataset(Dataset):
    def __init__(self, json_path, image_root, tokenizer):
        self.image_root = image_root
        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        self.tokenizer = tokenizer

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_root, sample['image'][0])  # list -> str
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        question = sample['question']
        encoding = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'pixel_values': image_tensor,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }

def get_dataloaders(tokenizer, batch_size=4):
    base = 'data/split'
    image_root = 'data/images'
    return {
        split: DataLoader(
            PuzzleWorldDataset(f'{base}/{split}.json', image_root, tokenizer),
            batch_size=batch_size,
            shuffle=(split == 'train')
        ) for split in ['train', 'val', 'test']
    }
