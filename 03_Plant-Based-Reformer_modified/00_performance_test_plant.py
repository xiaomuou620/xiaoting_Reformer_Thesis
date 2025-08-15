#!/usr/bin/env python3

import argparse, os, h5py
import torch
import torch.nn as nn
import transformers as T
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# CLI arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--h5file', required=True)
parser.add_argument('--model_path', required=True)
parser.add_argument('--tokenizer_dir', required=True)
parser.add_argument('--output_csv', required=True)
parser.add_argument('--roc_png', required=True)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()

# ----------------------------
# Dataset class
# ----------------------------
class SequenceDataset(Dataset):
    def __init__(self, h5_path, tokenizer, split='val'):
        self.tokenizer = tokenizer
        with h5py.File(h5_path, 'r') as h5:
            self.seq = h5[f'{split}_seq'][:]
            self.lab = h5[f'{split}_label'][:]
            self.code = h5[f'{split}_code_prefix'][:]

    def __len__(self): return len(self.lab)

    def __getitem__(self, idx):
        text = f"{self.code[idx].decode()} {self.seq[idx].decode()}"
        enc = self.tokenizer(text, truncation=True, return_tensors='pt')
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        label = torch.tensor(int(self.lab[idx]), dtype=torch.float32)
        return enc, label

def build_collate(tokenizer):
    def collate(batch):
        features, labels = zip(*batch)
        features = tokenizer.pad(features, return_tensors='pt')
        features.pop('token_type_ids', None)
        labels = torch.stack(labels)
        features['labels'] = labels
        return features
    return collate

# ----------------------------
# Model wrapper
# ----------------------------
class Bert4BinaryClassification(nn.Module):
    def __init__(self, encoder: T.PreTrainedModel, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        hid = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(hid, 1, bias=False), nn.Sigmoid())

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        pooled = out[:, 1:-1, :].mean(dim=1)
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        return logits

# ----------------------------
# Main
# ----------------------------
def main():
    device = torch.device(args.device)

    # Load tokenizer
    tokenizer = T.AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    with h5py.File(args.h5file, 'r') as h5:
        codes = {b.decode() for b in h5['val_code_prefix'][:]}
    tokenizer.add_tokens(list(codes))

    # Load model backbone (结构)
    encoder = T.AutoModel.from_pretrained('zhangtaolab/plant-dnabert-BPE', trust_remote_code=True)
    encoder.resize_token_embeddings(len(tokenizer))
    model = Bert4BinaryClassification(encoder)
    # 加载你保存的权重
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(device)
    model.eval()

    # Load data
    dataset = SequenceDataset(args.h5file, tokenizer, split='val')  # or split='test' if available
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=build_collate(tokenizer))

    # Inference
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits = model(**inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (logits >= 0.5).astype(int)

    # Save predictions
    df = pd.DataFrame({'label': labels, 'logit': logits, 'pred': preds})
    df.to_csv(args.output_csv, index=False)

    # Plot ROC
    fpr, tpr, _ = roc_curve(labels, logits)
    auc = roc_auc_score(labels, logits)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve ")
    plt.legend(loc="lower right")
    plt.savefig(args.roc_png)
    print(f"✅ ROC saved to {args.roc_png}")
    print(f"✅ Results saved to {args.output_csv}")
    print(f"✅ AUC: {auc:.4f}, Accuracy: {accuracy_score(labels, preds):.4f}")

if __name__ == '__main__':
    main()
