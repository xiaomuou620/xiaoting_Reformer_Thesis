import h5py
import numpy as np
import argparse
import os
# Disable tokenizers parallelism to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import random
import shutil
import time
import warnings
import math
import Bio.Seq
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
import torch.utils.data
import transformers as T
from transformers import get_scheduler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# Configure command-line arguments
parser = argparse.ArgumentParser(description='DNA-BERT Binary Classification Training')
parser.add_argument('-j',
                    '--workers',
                    default=1,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs',
                    default=6,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=2e-5,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-5,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training')
parser.add_argument("--lr_scheduler_type", type=str,
                    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                    default="cosine", help="The scheduler type to use")
parser.add_argument('--outdir', help='output directory')
parser.add_argument('--h5file', help='input h5 file path')
parser.add_argument('--device', nargs='+', help='list of GPU device IDs')
parser.add_argument('--resume', help='resume from checkpoint')


class SequenceDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for DNA sequences with barcode prefixes.
    Loads data from HDF5 file and tokenizes sequences using 3-mer tokens.
    """
    def __init__(self, h5file, tokenizer, max_length=512, train=False):
        # Open HDF5 file and select appropriate data split
        df = h5py.File(h5file, 'r')
        self.tokenizer = tokenizer
        self.max_length = max_length

        if train:
            self.sequence = df['trn_seq']  # Training sequences
            self.label = df['trn_label']   # Training labels
            self.barcode = df['trn_code_prefix']  # Training barcodes
        else:
            self.sequence = df['val_seq']  # Validation sequences
            self.label = df['val_label']   # Validation labels
            self.barcode = df['val_code_prefix']  # Validation barcodes

        self.n = len(self.label)
        self._df_handle = df  # Keep HDF5 file handle open during dataset lifetime

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # Decode sequence from bytes and get label
        ss = self.sequence[i].decode()
        label = int(self.label[i])
        
        # Convert sequence to 3-mer tokens
        ss = [ss[j:int(j+3)] for j in range(int(len(ss)-2))]  # 3-mer tokens
        
        # Prepare sequence with barcode prefix
        seq = [self.barcode[i].decode()]  # Start with barcode
        seq.extend(ss[:-1])  # Add 3-mer tokens (excluding last incomplete token)
        
        # Tokenize the sequence
        inputs = self.tokenizer(seq, is_split_into_words=True, add_special_tokens=True, return_tensors='pt')
        label = torch.tensor(label)
        barcode = self.barcode[i].decode()  # Return barcode for protein grouping
        
        return inputs['input_ids'], label, barcode


class Bert4BinaryClassification(nn.Module):
    """
    BERT-based model for binary classification of DNA sequences.
    Uses pre-trained DNA-BERT with additional classification layers.
    """
    def __init__(self, tokenizer):
        super(Bert4BinaryClassification, self).__init__()
        # Load pre-trained DNA-BERT model
        self.model = T.BertModel.from_pretrained("armheb/DNA_bert_3")
        self.model.resize_token_embeddings(292)  # Resize for additional barcode tokens
        
        # Classification layers
        hidden_size = 768
        self.dropout = nn.Dropout(0.2)
        self.lin = nn.Linear(hidden_size, 1, bias=False)  # Linear transformation for each position
        self.classification = nn.Linear(509, 1, bias=False)  # Final classification layer
        self.sigmoid = nn.Sigmoid()  # Output sigmoid for binary classification

    def forward(self, input_ids):
        # Get hidden states from BERT, excluding special tokens [CLS] and [SEP]
        hidden = self.model(input_ids=input_ids.squeeze(1)).last_hidden_state[:, 2:-1, :]
        hidden = self.dropout(hidden)
        
        # Apply linear transformation and final classification
        score = self.lin(hidden).squeeze()
        predict = self.classification(score)
        predict = self.sigmoid(predict)
        return predict


def main():
    args = parser.parse_args()
    assert args.outdir is not None

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('Deterministic CUDNN enabled due to seeding; may slow training.')

    gpus = args.device
    main_worker(gpus=gpus, args=args)


def main_worker(gpus, args):
    """Main training worker function."""
    # Initialize tokenizer
    tokenizer = T.BertTokenizer.from_pretrained("armheb/DNA_bert_3")

    # Add custom barcode tokens to tokenizer
    df = h5py.File(args.h5file, 'r')
    prefix_token = list(set([i.decode() for i in df['trn_code_prefix'][:]]))
    df.close()
    tokenizer.add_tokens(prefix_token)
    tokenizer.save_pretrained(args.outdir)
    
    # Initialize model
    model = Bert4BinaryClassification(tokenizer)

    # Load checkpoint if resuming training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint)

    print(model.model.config)
    print(model)

    # Move model to GPU and setup data parallel
    model.cuda()
    model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    # Setup optimizer with weight decay groups (no decay for bias and LayerNorm)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.95))

    # Setup loss function
    criterion = nn.BCELoss()

    # Enable cudnn benchmark for improved performance
    cudnn.benchmark = True
    
    # Create datasets
    train_dataset = SequenceDataset(args.h5file, tokenizer, train=True)
    val_dataset = SequenceDataset(args.h5file, tokenizer, train=False)

    PAD_TOKEN_ID = tokenizer.pad_token_id

    # Custom collate function that handles variable-length sequences and includes barcodes
    def collate_fn(x):
        input_ids = [ids.squeeze() for ids, _, _ in x]
        labels = [label for _, label, _ in x]
        barcodes = [bc for _, _, bc in x]
        
        # Pad sequences to same length
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, padding_value=PAD_TOKEN_ID)
        mask = (input_ids != PAD_TOKEN_ID).int()
        labels = torch.stack(labels)
        
        return {'input_ids': input_ids.T, 'attention_mask': mask.T, 'labels': labels, 'barcodes': barcodes}

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Setup learning rate scheduler
    num_update_steps_per_epoch = math.ceil(len(train_loader))
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=1e4,
        num_training_steps=max_train_steps
    )

    # Handle evaluation-only mode
    if args.evaluate:
        # Single evaluation run
        _ = validate(val_loader, model, criterion, args, epoch=args.start_epoch)
        return

    best_auc = 0.5

    # Setup long-format metrics logging (append mode)
    metrics_long_path = os.path.join(args.outdir, 'metrics_long.csv')
    if os.path.exists(metrics_long_path):
        os.remove(metrics_long_path)

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        # Training phase
        train_loss, train_acc, train_auc, train_group_df = train(
            train_loader, model, criterion, lr_scheduler, optimizer, epoch, args
        )

        # Validation phase
        val_loss, val_acc, val_auc, val_group_df = validate(
            val_loader, model, criterion, args, epoch
        )

        # Log epoch-level metrics in wide format (for backward compatibility)
        if epoch == args.start_epoch:
            log_cols = ['epoch', 'train_loss', 'train_acc', 'train_auc', 'val_loss', 'val_acc', 'val_auc']
            log_df = pd.DataFrame(columns=log_cols)
        
        # Create metrics row for current epoch
        row = pd.DataFrame({
            'epoch': [epoch],
            'train_loss': [train_loss],
            'train_acc': [train_acc],
            'train_auc': [train_auc],
            'val_loss': [val_loss],
            'val_acc': [val_acc],
            'val_auc': [val_auc],
        })
        
        # Append to training log
        if epoch == args.start_epoch:
            log_df = row
        else:
            log_df = pd.concat([pd.read_csv(os.path.join(args.outdir, 'training_log.csv')), row], ignore_index=True)
        log_df.to_csv(os.path.join(args.outdir, 'training_log.csv'), index=False)

        # Append long-format metrics (per-protein breakdown)
        train_group_df = train_group_df.assign(split='train')
        val_group_df = val_group_df.assign(split='val')
        metrics_long_epoch = pd.concat([train_group_df, val_group_df], ignore_index=True)
        
        # Append to long-format CSV
        if os.path.exists(metrics_long_path):
            metrics_long_epoch.to_csv(metrics_long_path, mode='a', header=False, index=False)
        else:
            metrics_long_epoch.to_csv(metrics_long_path, index=False)

        # Save model checkpoints
        is_best = val_auc > best_auc
        best_auc = max(val_auc, best_auc)
        torch.save(model.module.state_dict(), f'{args.outdir}/model_epoch{epoch}.bin')
        if is_best:
            torch.save(model.module.state_dict(), f'{args.outdir}/model_best.bin')


def train(train_loader, model, criterion, scheduler, optimizer, epoch, args):
    """
    One epoch of training with per-sample prediction collection for grouped metrics.
    """
    # Initialize metric tracking
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':6.2f')
    aucs = AverageMeter('AUC', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, accs, aucs],
                             prefix=f"Epoch: [{epoch}]")

    model.train()

    # Collect predictions for epoch-level grouped metrics
    all_logits = []
    all_labels = []
    all_barcodes = []

    end = time.time()
    for i, batch in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move data to GPU
        input_ids = batch['input_ids'].cuda(non_blocking=True)
        label = batch['labels'].cuda(non_blocking=True).view(-1).float()
        barcodes = batch['barcodes']

        # Forward pass
        logits = model(input_ids).view(-1)
        loss = criterion(logits, label)

        # Compute mini-batch metrics
        acc, auc = compute_metrics(logits.detach().cpu().squeeze(), label.detach().cpu().squeeze())
        losses.update(loss.item(), input_ids.size(0))
        accs.update(acc, input_ids.size(0))
        aucs.update(auc, input_ids.size(0))

        # Accumulate predictions for epoch-level grouped metrics
        all_logits.extend(logits.detach().cpu().numpy())
        all_labels.extend(label.detach().cpu().numpy())
        all_barcodes.extend(barcodes)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print progress
        if i % args.print_freq == 0:
            progress.display(i)

    # Save per-sample predictions for this epoch (training)
    train_pred_df = pd.DataFrame({
        'barcode': all_barcodes,
        'true_label': all_labels,
        'pred_score': all_logits,
        'pred_label': [int(x >= 0.5) for x in all_logits],
    })
    train_pred_df.to_csv(os.path.join(args.outdir, f'train_preds_epoch{epoch}.csv'), index=False)

    # Compute grouped metrics by protein and save
    train_group_df = _group_metrics_df(train_pred_df, epoch)
    train_group_df.to_csv(os.path.join(args.outdir, f'train_metrics_by_protein_epoch{epoch}.csv'), index=False)

    return losses.avg, accs.avg, aucs.avg, train_group_df


def validate(val_loader, model, criterion, args, epoch):
    """
    One epoch of validation with per-sample prediction collection.
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':6.2f')
    aucs = AverageMeter('AUC', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, accs, aucs],
                             prefix='Validation: ')

    model.eval()

    all_logits = []
    all_labels = []
    all_barcodes = []

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].cuda(non_blocking=True)
            label = batch['labels'].cuda(non_blocking=True).view(-1).float()
            barcodes = batch['barcodes']

            logits = model(input_ids).view(-1)
            loss = criterion(logits, label)

            acc, auc = compute_metrics(logits.detach().cpu().squeeze(), label.detach().cpu().squeeze())
            losses.update(loss.item(), input_ids.size(0))
            accs.update(acc, input_ids.size(0))
            aucs.update(auc, input_ids.size(0))

            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_barcodes.extend(barcodes)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    print(f' * Loss {losses.avg:.3f}')

    # Save per-sample predictions (validation)
    pred_df = pd.DataFrame({
        'barcode': all_barcodes,
        'true_label': all_labels,
        'pred_score': all_logits,
        'pred_label': [int(i >= 0.5) for i in all_logits]
    })
    pred_df.to_csv(os.path.join(args.outdir, f'val_preds_epoch{epoch}.csv'), index=False)

    # Compute and save grouped metrics (validation)
    group_df = _group_metrics_df(pred_df, epoch)
    group_df.to_csv(os.path.join(args.outdir, f'val_metrics_by_protein_epoch{epoch}.csv'), index=False)

    return losses.avg, accs.avg, aucs.avg, group_df


def _group_metrics_df(pred_df, epoch):
    """
    Helper function: given per-sample predictions DataFrame (barcode, true_label, pred_score, pred_label),
    return grouped metrics DataFrame with rows for ALL samples + each protein.
    """
    # Extract protein name from barcode (adjust regex if barcode format changes)
    pred_df = pred_df.copy()
    pred_df['protein'] = pred_df['barcode'].str.extract(r'^([A-Za-z0-9]+)')

    stats = []
    proteins = ['ALL'] + sorted(pred_df['protein'].dropna().unique().tolist())
    
    for prot in proteins:
        if prot == 'ALL':
            df_sub = pred_df  # Use all data for overall metrics
        else:
            df_sub = pred_df[pred_df['protein'] == prot]  # Filter by protein

        y_true = df_sub['true_label'].values
        y_pred = df_sub['pred_label'].values
        y_score = df_sub['pred_score'].values

        # Compute metrics
        if len(np.unique(y_true)) < 2:
            auc = float('nan')  # Cannot compute AUC with single class
        else:
            auc = roc_auc_score(y_true, y_score)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        stats.append({
            'epoch': epoch,
            'protein': prot,
            'loss': float('nan'),  # Epoch-level split loss handled outside; left NaN here
            'acc': acc,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(df_sub)
        })

    return pd.DataFrame(stats)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save model checkpoint and copy best model if needed."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value of metrics."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """Display progress during training and validation."""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs."""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_metrics(outputs, labels):
    """
    Compute accuracy and AUC metrics from model outputs and true labels.
    
    Args:
        outputs: Model predictions as probabilities (0-1)
        labels: True binary labels
        
    Returns:
        tuple: (accuracy, auc) scores
    """
    predicted_labels = (outputs >= 0.5).float()
    accuracy = accuracy_score(labels.cpu(), predicted_labels.cpu())
    try:
        auc = roc_auc_score(labels.cpu(), outputs.cpu())
    except ValueError:
        # Handle case where all labels are the same class
        auc = float('nan')
    return accuracy, auc


if __name__ == '__main__':
    main()