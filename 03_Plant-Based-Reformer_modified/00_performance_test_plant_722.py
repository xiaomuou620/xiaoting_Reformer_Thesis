#!/usr/bin/env python3
"""
DNA Sequence Binary Classification Inference Script

This script performs inference on a trained BERT-based DNA sequence classification model,
evaluates performance metrics both overall and per protein type, and generates 
comprehensive analysis reports including ROC curves and detailed predictions.
"""

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
# Command Line Interface Arguments
# ----------------------------
parser = argparse.ArgumentParser(description='Perform inference on DNA sequence binary classification model')
parser.add_argument('--h5file', required=True, help='Path to HDF5 data file containing sequences and labels')
parser.add_argument('--model_path', required=True, help='Path to trained model weights file')
parser.add_argument('--tokenizer_dir', required=True, help='Directory containing tokenizer files')
parser.add_argument('--output_csv', required=True, help='Output CSV file path for predictions')
parser.add_argument('--roc_png', required=True, help='Output PNG file path for ROC curve plot')
parser.add_argument('--device', default='cuda:0', help='Computing device (cuda:0, cuda:1, cpu, etc.)')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for model inference')
parser.add_argument('--model_name', default='zhangtaolab/plant-dnabert-BPE', help='HuggingFace model name or local path')
args = parser.parse_args()

# ----------------------------
# Dataset class for DNA sequence processing
# ----------------------------
class SequenceDataset(Dataset):
    """
    PyTorch Dataset class for loading and processing DNA sequences from HDF5 files.
    Combines protein code prefixes with DNA sequences for tokenization.
    """
    def __init__(self, h5_path, tokenizer, split='val'):
        """
        Initialize the dataset by loading sequences, labels, and protein codes.
        
        Args:
            h5_path: Path to the HDF5 file containing the data
            tokenizer: Pre-trained tokenizer for sequence encoding
            split: Data split to load ('val', 'test', 'train')
        """
        self.tokenizer = tokenizer
        with h5py.File(h5_path, 'r') as h5:
            self.seq = h5[f'{split}_seq'][:]      # DNA sequences
            self.lab = h5[f'{split}_label'][:]    # Binary classification labels
            self.code = h5[f'{split}_code_prefix'][:]  # Protein identifier codes

    def __len__(self): 
        """Return the total number of samples in the dataset."""
        return len(self.lab)

    def __getitem__(self, idx):
        """
        Retrieve and process a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (tokenized_sequence, label)
        """
        # Combine protein code with DNA sequence for contextual information
        text = f"{self.code[idx].decode()} {self.seq[idx].decode()}"
        
        # Tokenize the combined text with truncation for consistent length
        enc = self.tokenizer(text, truncation=True, return_tensors='pt')
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        
        # Convert label to float tensor for binary classification
        label = torch.tensor(int(self.lab[idx]), dtype=torch.float32)
        return enc, label

def build_collate(tokenizer):
    """
    Build a collate function for the DataLoader to handle batching of variable-length sequences.
    
    Args:
        tokenizer: The tokenizer used for padding sequences
        
    Returns:
        Collate function that pads sequences and prepares batch tensors
    """
    def collate(batch):
        """
        Collate function to process a batch of samples.
        Pads sequences to the same length and stacks labels.
        """
        features, labels = zip(*batch)
        # Pad all sequences in the batch to the same length
        features = tokenizer.pad(features, return_tensors='pt')
        # Remove token_type_ids if present (not needed for this model)
        features.pop('token_type_ids', None)
        # Stack all labels into a single tensor
        labels = torch.stack(labels)
        features['labels'] = labels
        return features
    return collate

# ----------------------------
# BERT-based binary classification model
# ----------------------------
class Bert4BinaryClassification(nn.Module):
    """
    Binary classification model based on pre-trained BERT architecture.
    Uses mean pooling of hidden states and a sigmoid classifier head.
    """
    def __init__(self, encoder: T.PreTrainedModel, dropout=0.2):
        """
        Initialize the classification model.
        
        Args:
            encoder: Pre-trained transformer model (BERT-like)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.encoder = encoder
        hid = encoder.config.hidden_size  # Get hidden dimension from encoder config
        self.dropout = nn.Dropout(dropout)
        # Classification head: Linear layer followed by Sigmoid for binary classification
        self.classifier = nn.Sequential(nn.Linear(hid, 1, bias=False), nn.Sigmoid())

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input sequences
            attention_mask: Mask to ignore padded tokens
            
        Returns:
            Binary classification probabilities (0-1)
        """
        # Get hidden representations from the encoder
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        
        # Mean pooling over sequence dimension, excluding special tokens [CLS] and [SEP]
        pooled = out[:, 1:-1, :].mean(dim=1)
        
        # Apply dropout and classification head
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        return logits

# ----------------------------
# Main inference and evaluation pipeline
# ----------------------------
def main():
    """
    Main function orchestrating the entire inference and evaluation process.
    """
    device = torch.device(args.device)

    # Load and configure tokenizer
    print(" Loading tokenizer...")
    tokenizer = T.AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    
    # Add protein codes to tokenizer vocabulary
    with h5py.File(args.h5file, 'r') as h5:
        codes = {b.decode() for b in h5['val_code_prefix'][:]}
    tokenizer.add_tokens(list(codes))
    print(f" Added {len(codes)} protein codes to tokenizer vocabulary")

    # Load model architecture and weights
    print(" Loading model...")
    encoder = T.AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    encoder.resize_token_embeddings(len(tokenizer))  # Resize for new protein codes
    model = Bert4BinaryClassification(encoder)
    
    # Load trained model weights
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(device)
    model.eval()  # Set to evaluation mode
    print(f" Model loaded and moved to {device}")

    # Prepare dataset and dataloader
    print(" Preparing data...")
    dataset = SequenceDataset(args.h5file, tokenizer, split='val')  # Use 'test' if available
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=build_collate(tokenizer))
    print(f" Dataset loaded: {len(dataset)} samples")

    # Perform model inference
    print(" Running inference...")
    all_logits = []
    all_labels = []
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_idx, batch in enumerate(loader):
            # Move batch to device
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            logits = model(**inputs)
            
            # Collect predictions and labels
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
            # Progress indicator
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches")

    # Concatenate all results
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (logits >= 0.5).astype(int)  # Binary predictions using 0.5 threshold

    # Create results DataFrame
    df = pd.DataFrame({'label': labels, 'logit': logits, 'pred': preds})

    # Add protein type information
    with h5py.File(args.h5file, 'r') as h5:
        codes = [b.decode() for b in h5['val_code_prefix'][:]]
    df['code_prefix'] = codes

    # Save overall predictions
    df.to_csv(args.output_csv, index=False)

    # Generate and save ROC curve
    print("📈 Generating ROC curve...")
    fpr, tpr, _ = roc_curve(labels, logits)
    auc = roc_auc_score(labels, logits)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal reference line
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.roc_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" ROC curve saved to {args.roc_png}")
    print(f" Overall results saved to {args.output_csv}")
    print(f" Overall AUC: {auc:.4f}, Accuracy: {accuracy_score(labels, preds):.4f}")

    # --------- Detailed analysis for each protein type ---------
    print("\n🔬 Analyzing per-protein performance...")
    protein_metrics = []
    
    for protein in sorted(set(df['code_prefix'])):
        sub_df = df[df['code_prefix'] == protein]
        if len(sub_df) < 2: 
            continue  # Need at least 2 samples for meaningful metrics
            
        y_true = sub_df['label']
        y_score = sub_df['logit']
        y_pred = sub_df['pred']
        
        # Calculate basic metrics
        acc = accuracy_score(y_true, y_pred)
        
        # Calculate AUC with error handling
        try:
            auc_p = roc_auc_score(y_true, y_score)
        except:
            auc_p = 0.0  # Handle cases where AUC cannot be calculated (e.g., all same label)
            
        # Calculate precision, recall, and F1 score
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Store metrics for this protein
        protein_metrics.append({
            'protein': protein,
            'count': len(sub_df),
            'accuracy': acc,
            'auc': auc_p,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Save detailed predictions for each protein
        protein_detail_path = os.path.join(os.path.dirname(args.output_csv), f'protein_{protein}_details.csv')
        sub_df.to_csv(protein_detail_path, index=False)

    # Save summary metrics for all protein types
    metrics_df = pd.DataFrame(protein_metrics)
    metrics_df = metrics_df.sort_values('protein')
    protein_summary_path = os.path.join(os.path.dirname(args.output_csv), 'protein_metrics_summary.csv')
    metrics_df.to_csv(protein_summary_path, index=False)
    
    print(f" Per-protein metrics saved to protein_metrics_summary.csv")
    print(f" Per-protein details saved to protein_*_details.csv")

    # Overall performance analysis
    print("\n📊 Computing overall performance metrics...")
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    # Calculate comprehensive overall metrics
    overall_metrics = {
        'accuracy': accuracy_score(labels, preds),
        'auc': roc_auc_score(labels, logits) if len(set(labels)) > 1 else 0.0,
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'tp': int(((labels == 1) & (preds == 1)).sum()),  # True Positives
        'tn': int(((labels == 0) & (preds == 0)).sum()),  # True Negatives
        'fp': int(((labels == 0) & (preds == 1)).sum()),  # False Positives
        'fn': int(((labels == 1) & (preds == 0)).sum()),  # False Negatives
        'total': int(len(labels))
    }
    
    # Save overall metrics to CSV
    overall_metrics_df = pd.DataFrame([overall_metrics])
    overall_metrics_path = os.path.join(os.path.dirname(args.output_csv), 'overall_metrics.csv')
    overall_metrics_df.to_csv(overall_metrics_path, index=False)

    # Display results summary
    print("\n" + "="*60)
    print(" OVERALL PERFORMANCE METRICS")
    print("="*60)
    print(overall_metrics_df.T)

    # Display confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"\n Confusion Matrix:")
    print("     Predicted")
    print("       0    1")
    print(f"Actual 0  {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"       1  {cm[1,0]:4d} {cm[1,1]:4d}")
    
    print(f"\n Analysis Summary:")
    print(f"   • Total samples: {len(labels)}")
    print(f"   • Protein types: {len(set(df['code_prefix']))}")
    print(f"   • Best performing metric: AUC = {auc:.4f}")
    print(f"   • Classification threshold: 0.5")

if __name__ == '__main__':
    main()