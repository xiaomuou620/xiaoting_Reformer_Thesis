import h5py, torch, argparse, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import seaborn as sns

class SequenceDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading DNA sequences from HDF5 file
    Handles tokenization and 3-mer generation for BERT-based models
    """
    def __init__(self, h5file, tokenizer, split):
        """
        Initialize dataset
        Args:
            h5file: Path to HDF5 file containing sequences and labels
            tokenizer: BERT tokenizer for sequence encoding
            split: Data split name ('train', 'test', 'val')
        """
        df = h5py.File(h5file, 'r')
        self.tokenizer = tokenizer
        self.sequence = df[f'{split}_seq']
        self.label = df[f'{split}_label']
        self.barcode = df[f'{split}_code_prefix']
        self.n = len(self.label)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        """
        Get a single sample with 3-mer tokenization
        Returns tokenized sequence, label, and barcode identifier
        """
        ss = self.sequence[i].decode()
        label = int(self.label[i])
        ss = [ss[j:j+3] for j in range(len(ss)-2)]  # Generate 3-mer sequences
        seq = [self.barcode[i].decode()]
        seq.extend(ss[:-1])
        inputs = self.tokenizer(seq, is_split_into_words=True, add_special_tokens=True, return_tensors='pt') 
        return inputs['input_ids'], torch.tensor(label), self.barcode[i].decode()

class Bert4BinaryClassification(torch.nn.Module):
    """
    BERT-based binary classification model for DNA sequences
    Uses pre-trained DNA-BERT model with custom classification head
    """
    def __init__(self, tokenizer):
        super().__init__()
        # Load pre-trained DNA-BERT model
        self.model = BertModel.from_pretrained("armheb/DNA_bert_3")
        self.model.resize_token_embeddings(292)  # Resize for DNA vocabulary
        self.dropout = torch.nn.Dropout(0.2)
        self.lin = torch.nn.Linear(768, 1, bias=False)  # Hidden state projection
        self.classification = torch.nn.Linear(509, 1, bias=False)  # Final classification layer
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, input_ids):
        """
        Forward pass through the model
        Args:
            input_ids: Tokenized input sequences
        Returns:
            Probability scores for binary classification
        """
        # Get BERT hidden states, excluding special tokens
        hidden = self.model(input_ids=input_ids.squeeze(1)).last_hidden_state[:, 2:-1, :]
        hidden = self.dropout(hidden)
        score = self.lin(hidden).squeeze()
        predict = self.classification(score)
        predict = self.sigmoid(predict)
        return predict

def collate_fn(x, pad_token_id):
    """
    Collate function for DataLoader to handle variable length sequences
    Pads sequences to same length and creates attention masks
    """
    input_ids = [ids.squeeze() for ids, _, _ in x]
    labels = [label for _, label, _ in x]
    names = [name for _, _, name in x]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, padding_value=pad_token_id)
    mask = (input_ids != pad_token_id).int()
    labels = torch.stack(labels)
    return {'input_ids': input_ids.T, 'attention_mask': mask.T, 'labels': labels, 'names': names}

def compute_metrics(outputs, labels):
    """
    Compute basic evaluation metrics (accuracy and AUC)
    Args:
        outputs: Model prediction probabilities
        labels: Ground truth labels
    Returns:
        accuracy, AUC scores
    """
    predicted_labels = (outputs >= 0.5).float()
    acc = accuracy_score(labels.cpu(), predicted_labels.cpu())
    auc = roc_auc_score(labels.cpu(), outputs.cpu())
    return acc, auc

def plot_roc(labels, probs, output_path):
    """
    Generate and save ROC curve plot
    Args:
        labels: True labels
        probs: Predicted probabilities
        output_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_distribution(labels, probs, output_path):
    """Generate prediction score distribution plot"""
    plt.figure(figsize=(10, 6))
    
    # Plot score distributions for positive and negative samples separately
    pos_scores = np.array(probs)[np.array(labels) == 1]
    neg_scores = np.array(probs)[np.array(labels) == 0]
    
    plt.hist(neg_scores, bins=50, alpha=0.7, label='Negative (Label=0)', color='red', density=True)
    plt.hist(pos_scores, bins=50, alpha=0.7, label='Positive (Label=1)', color='blue', density=True)
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(labels, probs, output_path, threshold=0.5):
    """Generate confusion matrix heatmap"""
    pred_labels = (np.array(probs) >= threshold).astype(int)
    cm = confusion_matrix(labels, pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Predicted 0', 'Predicted 1'],
               yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix (Threshold={threshold})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm

def calculate_detailed_metrics(labels, probs, threshold=0.5):
    """
    Calculate comprehensive evaluation metrics
    Args:
        labels: True labels
        probs: Predicted probabilities
        threshold: Classification threshold
    Returns:
        Dictionary containing all metrics
    """
    pred_labels = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)
    
    # Basic metrics
    accuracy = np.mean(pred_labels == labels)
    
    # Confusion matrix elements
    tn = np.sum((pred_labels == 0) & (labels == 0))
    tp = np.sum((pred_labels == 1) & (labels == 1))
    fn = np.sum((pred_labels == 0) & (labels == 1))
    fp = np.sum((pred_labels == 1) & (labels == 0))
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC score
    auc = roc_auc_score(labels, probs)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'auc': auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def evaluate_split(model, dataloader, device, pad_token_id, split_name, output_dir):
    """
    Evaluate model on a specific data split and save results
    Args:
        model: Trained model to evaluate
        dataloader: DataLoader for the split
        device: Computing device (CPU/GPU)
        pad_token_id: Padding token ID
        split_name: Name of the split being evaluated
        output_dir: Directory to save outputs
    Returns:
        Dictionary of detailed metrics
    """
    model.eval()
    all_probs, all_labels, all_names = [], [], []
    loss_fn = torch.nn.BCELoss()
    all_loss = []

    # Run inference on all batches
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device).float()
            names = batch['names']
            outputs = model(input_ids).view(-1)
            loss = loss_fn(outputs, labels)
            all_loss.append(loss.item())
            all_probs.extend(outputs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_names.extend(names)

    # Calculate metrics
    acc, auc = compute_metrics(torch.tensor(all_probs), torch.tensor(all_labels))
    avg_loss = sum(all_loss) / len(all_loss)
    detailed_metrics = calculate_detailed_metrics(all_labels, all_probs)

    # Print summary results
    print(f"📊 {split_name.upper()} - Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print(f"   Precision: {detailed_metrics['precision']:.4f} | Recall: {detailed_metrics['recall']:.4f} | F1-Score: {detailed_metrics['f1_score']:.4f}")

    # Save overall prediction results CSV
    df = pd.DataFrame({
        'barcode': all_names, 
        'true_label': all_labels, 
        'pred_score': all_probs,
        'pred_label': (np.array(all_probs) >= 0.5).astype(int)
    })
    csv_path = os.path.join(output_dir, f"{split_name}_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 Predictions saved to {csv_path}")
    
    # Save overall detailed metrics
    metrics_df = pd.DataFrame([detailed_metrics])
    metrics_path = os.path.join(output_dir, f"{split_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"📋 Detailed metrics saved to {metrics_path}")

    # Calculate per-protein metrics
    per_protein_metrics = []
    for protein, group_df in df.groupby('barcode'):
        metrics = calculate_detailed_metrics(group_df['true_label'].values, group_df['pred_score'].values)
        metrics['barcode'] = protein
        per_protein_metrics.append(metrics)

    per_protein_df = pd.DataFrame(per_protein_metrics)
    per_protein_path = os.path.join(output_dir, f"{split_name}_per_protein_metrics.csv")
    per_protein_df.to_csv(per_protein_path, index=False)
    print(f"📋 Per-protein metrics saved to {per_protein_path}")
    
    return detailed_metrics

def generate_summary_report(all_metrics, output_dir):
    """Generate comprehensive summary analysis report"""
    # Create metrics comparison table
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df = metrics_df.round(4)
    
    # Save metrics summary
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    metrics_df.to_csv(summary_path)
    print(f"📋 Evaluation summary saved to {summary_path}")
    
    # Generate metrics comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink', 'lightsalmon']
    
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes):
            ax = axes[i]
            splits = list(all_metrics.keys())  # Dynamically get all split names
            values = [all_metrics[split][metric] for split in splits]
            bars = ax.bar(splits, values, color=colors[i], alpha=0.7)

            ax.set_title(f'{metric.capitalize().replace("_", " ")}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add values on top of bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Hide extra subplots
    for i in range(len(metrics_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Metrics comparison plot saved to {comparison_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    print(metrics_df)
    print(f"\n🎯 Best performing split by AUC: {metrics_df['auc'].idxmax()}")
    print(f"🎯 Highest AUC: {metrics_df['auc'].max():.4f}")
    print(f"🎯 Best performing split by F1-Score: {metrics_df['f1_score'].idxmax()}")
    print(f"🎯 Highest F1-Score: {metrics_df['f1_score'].max():.4f}")

def main():
    """
    Main evaluation function
    Parses arguments, loads model and data, runs evaluation, and generates reports
    """
    parser = argparse.ArgumentParser(description='Evaluate DNA sequence binary classification model')
    parser.add_argument('--h5file', required=True, help='Path to HDF5 data file')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--tokenizer_dir', required=True, help='Path to tokenizer directory')
    parser.add_argument('--output_dir', default='evaluation_outputs', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--device', default='cuda:0', help='Device to use for evaluation')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_dir)
    pad_token_id = tokenizer.pad_token_id

    # Load model
    model = Bert4BinaryClassification(tokenizer)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(args.device)

    # Evaluate each dataset split
    all_metrics = {}
    for split in ['test']:
        dataset = SequenceDataset(args.h5file, tokenizer, split)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=lambda x: collate_fn(x, pad_token_id))
        split_metrics = evaluate_split(model, dataloader, args.device, pad_token_id, split, args.output_dir)
        all_metrics[split] = split_metrics

    # Generate summary report
    generate_summary_report(all_metrics, args.output_dir)

if __name__ == '__main__':
    main()