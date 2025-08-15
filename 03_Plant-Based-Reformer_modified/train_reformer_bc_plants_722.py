#!/usr/bin/env python3

#modification of train_reformer_bc.py that can use different DNABERT-style encoders (foundation models)
#defaults to plant-DNABERT-BPE
#binary classification

import os, random, time, math, warnings, argparse, shutil, h5py
import pandas as pd
import numpy as np
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import transformers as T
from transformers import get_scheduler

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Binary classification on DNA with DNABERT‑style encoder")
parser.add_argument('--workers', '-j', default=2, type=int,
                    help='data‑loading workers')
parser.add_argument('--epochs', default=6, type=int)
parser.add_argument('--batch-size', '-b', default=64, type=int)
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
parser.add_argument('--wd', '--weight-decay', dest='weight_decay',
                    default=1e-5, type=float)
parser.add_argument('--print-freq', '-p', default=10, type=int)
parser.add_argument('--lr_scheduler_type', default='cosine',
                    choices=["linear", "cosine", "cosine_with_restarts",
                             "polynomial", "constant", "constant_with_warmup"])
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--seed', type=int) #useful for training using different seeds to check the stability
parser.add_argument('--outdir', required=True, help='output directory')
parser.add_argument('--h5file', required=True, help='input h5')
parser.add_argument('--device', nargs='+', required=True, help='GPU ids, can use in DataParallel e.g. 0 1 2')
parser.add_argument('--resume', help='checkpoint to resume')
parser.add_argument('--model_name',
                    default='zhangtaolab/plant-dnabert-BPE',
                    help='HF model or local dir (default: plant DNABERT)')
                    
#can try different models https://huggingface.co/collections/zhangtaolab/plant-foundation-models-668514bd5c384b679f9bdf17

#-----------------------------------------------------------------------------
#Process dataset -- convert the HDF5 data into tensors that can be used in transformer

##Each item in the HDF5 has: 
### seq -- sequence
### lab -- binary 0/1 according to protein binding
### code -- for multi-tasking e.g. RBP_X
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, tokenizer, split):
        self.tokenizer = tokenizer
        with h5py.File(h5_path) as h5:
            if split == 'train':
                self.seq  = h5['trn_seq'][:]
                self.lab  = h5['trn_label'][:]
                self.code = h5['trn_code_prefix'][:]
            else:
                self.seq  = h5['val_seq'][:]
                self.lab  = h5['val_label'][:]
                self.code = h5['val_code_prefix'][:]

    def __len__(self): return len(self.lab)

####this part prefixes the barcode in fromt of the raw sequence, perfors BPE, adds [CLS]/[SEP] and returns tensors
    def __getitem__(self, idx):
        # prepend barcode token; then let tokenizer handle BPE & special tokens
        text = f"{self.code[idx].decode()} {self.seq[idx].decode()}"
        enc  = self.tokenizer(text, truncation=True, return_tensors='pt')
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        label = torch.tensor(int(self.lab[idx]), dtype=torch.float32)
        barcode = self.code[idx].decode()  # Return barcode for grouping
        return enc, label, barcode

#-----------------------------------------------------------------------------
#Model wrapper -- classification head on top of encoder (Plant-DNABERT-BPE model)
#classifier: Dropout -> Linear -> Sigmoid
class Bert4BinaryClassification(nn.Module):
    def __init__(self, encoder: T.PreTrainedModel, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        hid = encoder.config.hidden_size #hidden size is normally 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(hid, 1, bias=False),
                                        nn.Sigmoid())

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        # drop CLS & SEP, mean‑pool remaining tokens
        pooled = out[:, 1:-1, :].mean(dim=1)
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        return logits

#-----------------------------------------------------------------------------
class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val, self.sum, self.count = val, self.sum + val * n, self.count + n
        self.avg = self.sum / self.count
    def __str__(self):
        fmt = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmt.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, N, meters, prefix=""): 
        self.N, self.meters, self.prefix = N, meters, prefix
    def display(self, batch):
        num = f"[{batch:>{len(str(self.N))}}/{self.N}]"
        print('\t'.join([self.prefix + num] + [str(m) for m in self.meters]))

def compute_metrics(logits, labels):
    pred = (logits >= .5).float()
    acc  = accuracy_score(labels.cpu(), pred.cpu())
    try:
        auc  = roc_auc_score(labels.cpu(), logits.cpu())
    except ValueError:
        auc = float('nan')
    return acc, auc

#-----------------------------------------------------------------------------
#Collate
def build_collate(tokenizer):
    def collate(batch):
        features, labels, barcodes = zip(*batch)
        features = tokenizer.pad(features, return_tensors='pt')
        #drop token_type_ids so the wrapper never sees it
        features.pop('token_type_ids', None)
        labels = torch.stack(labels)
        features['labels'] = labels
        features['barcodes'] = barcodes
        return features
    return collate


def _group_metrics_df(pred_df, epoch):
    """Helper: given per-sample pred_df(barcode,true_label,pred_score,pred_label),
    return grouped metrics DataFrame with rows for ALL + each protein."""
    # protein name from barcode; adjust regex if barcode format changes
    pred_df = pred_df.copy()
    pred_df['protein'] = pred_df['barcode'].str.extract(r'^([A-Za-z0-9]+)')

    stats = []
    proteins = ['ALL'] + sorted(pred_df['protein'].dropna().unique().tolist())
    for prot in proteins:
        if prot == 'ALL':
            df_sub = pred_df
        else:
            df_sub = pred_df[pred_df['protein'] == prot]

        y_true = df_sub['true_label'].values
        y_pred = df_sub['pred_label'].values
        y_score = df_sub['pred_score'].values

        # metrics
        if len(np.unique(y_true)) < 2:
            auc = float('nan')  # can't compute
        else:
            auc = roc_auc_score(y_true, y_score)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        stats.append({
            'epoch': epoch,
            'protein': prot,
            'loss': float('nan'),  # epoch-level split loss handled outside; left NaN here
            'acc': acc,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(df_sub)
        })

    return pd.DataFrame(stats)


#-----------------------------------------------------------------------------
#Main worker
def main_worker(gpus, args):
    # tokenizer ---------------------------------------------------------------
    tok = T.AutoTokenizer.from_pretrained(args.model_name,
                                          trust_remote_code=True)
    # add barcode tokens
    with h5py.File(args.h5file) as h5:
        codes = {b.decode() for b in h5['trn_code_prefix'][:]}
    tok.add_tokens(list(codes))
    os.makedirs(args.outdir, exist_ok=True)
    tok.save_pretrained(args.outdir)

    # encoder -----------------------------------------------------------------
    base = T.AutoModel.from_pretrained(args.model_name,
                                       trust_remote_code=True)
    base.resize_token_embeddings(len(tok))
    model = Bert4BinaryClassification(base)

    base.config.to_json_file(os.path.join(args.outdir, "config.json"))

    # resume?
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))

    model.cuda()
    model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    # optimiser & scheduler ---------------------------------------------------
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = AdamW(grouped, lr=args.lr, betas=(0.9, 0.95))
    criterion = nn.BCELoss()

    # datasets & loaders ------------------------------------------------------
    collate_fn = build_collate(tok)
    trn_ds = SequenceDataset(args.h5file, tok, 'train')
    val_ds = SequenceDataset(args.h5file, tok, 'val')

    trn_ld = torch.utils.data.DataLoader(
        trn_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    val_ld = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True, collate_fn=collate_fn)

    steps_per_epoch = math.ceil(len(trn_ld))
    max_steps = args.epochs * steps_per_epoch
    scheduler = get_scheduler(args.lr_scheduler_type, optim,
                              num_warmup_steps=1_000, num_training_steps=max_steps)

    # -------------------------------------------------------------------------
    cudnn.benchmark = True
    best_auc = 0.5

    if args.evaluate:
        _ = validate(val_ld, model, criterion, args, epoch=args.start_epoch)
        return

    # Create long-format metrics log (append mode)
    metrics_long_path = os.path.join(args.outdir, 'metrics_long.csv')
    if os.path.exists(metrics_long_path):
        os.remove(metrics_long_path)

    for epoch in range(args.start_epoch, args.epochs):
        # --- Train ---
        train_loss, train_acc, train_auc, train_group_df = train_epoch(
            trn_ld, model, criterion, scheduler, optim, epoch, args
        )

        # --- Validate ---
        val_loss, val_acc, val_auc, val_group_df = validate(
            val_ld, model, criterion, args, epoch
        )

        # --- Wide format row (optional quick look) ---
        # (keep backward compatibility if you want training_log.csv)
        if epoch == args.start_epoch:
            log_cols = ['epoch', 'train_loss', 'train_acc', 'train_auc', 'val_loss', 'val_acc', 'val_auc']
            log_df = pd.DataFrame(columns=log_cols)
        # append row
        row = pd.DataFrame({
            'epoch': [epoch],
            'train_loss': [train_loss],
            'train_acc': [train_acc],
            'train_auc': [train_auc],
            'val_loss': [val_loss],
            'val_acc': [val_acc],
            'val_auc': [val_auc],
        })
        if epoch == args.start_epoch:
            log_df = row
        else:
            log_df = pd.concat([pd.read_csv(os.path.join(args.outdir, 'training_log.csv')), row], ignore_index=True)
        log_df.to_csv(os.path.join(args.outdir, 'training_log.csv'), index=False)

        # --- Append long-format metrics (split × protein) ---
        train_group_df = train_group_df.assign(split='train')
        val_group_df = val_group_df.assign(split='val')
        metrics_long_epoch = pd.concat([train_group_df, val_group_df], ignore_index=True)
        # append to CSV
        if os.path.exists(metrics_long_path):
            metrics_long_epoch.to_csv(metrics_long_path, mode='a', header=False, index=False)
        else:
            metrics_long_epoch.to_csv(metrics_long_path, index=False)

        # --- Save models ---
        is_best = val_auc > best_auc
        best_auc = max(val_auc, best_auc)
        torch.save(model.module.state_dict(), f'{args.outdir}/model_epoch{epoch}.bin')
        if is_best:
            torch.save(model.module.state_dict(), f'{args.outdir}/model_best.bin')

#-----------------------------------------------------------------------------
#Train & validate loops
def train_epoch(loader, model, crit, sched, optim, epoch, args):
    """One epoch of training + collect per-sample preds for grouped metrics."""
    t = AverageMeter('Time', ':6.3f')
    d = AverageMeter('Data', ':6.3f')
    l = AverageMeter('Loss', ':.4e')
    a = AverageMeter('Acc',  ':6.2f')
    u = AverageMeter('AUC',  ':6.2f')
    prog = ProgressMeter(len(loader), [t, d, l, a, u],
                         prefix=f"Epoch: [{epoch}]")

    model.train()

    all_logits = []
    all_labels = []
    all_barcodes = []

    end = time.time()
    for i, batch in enumerate(loader):
        d.update(time.time() - end)
        inp = {k: v.cuda(non_blocking=True) for k, v in batch.items()
               if k not in ['labels', 'barcodes']}
        lab = batch['labels'].cuda(non_blocking=True)
        barcodes = batch['barcodes']

        log = model(**inp)
        loss = crit(log, lab)

        acc, auc = compute_metrics(log.detach().cpu(), lab.detach().cpu())
        l.update(loss.item(), lab.size(0))
        a.update(acc, lab.size(0))
        u.update(auc, lab.size(0))

        # accumulate for epoch-level grouped metrics
        all_logits.extend(log.detach().cpu().numpy())
        all_labels.extend(lab.detach().cpu().numpy())
        all_barcodes.extend(barcodes)

        optim.zero_grad()
        loss.backward()
        optim.step()
        sched.step()

        t.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            prog.display(i)

    # --- save per-sample preds for this epoch (train) ---
    train_pred_df = pd.DataFrame({
        'barcode': all_barcodes,
        'true_label': all_labels,
        'pred_score': all_logits,
        'pred_label': [int(x >= 0.5) for x in all_logits],
    })
    train_pred_df.to_csv(os.path.join(args.outdir, f'train_preds_epoch{epoch}.csv'), index=False)

    # Add protein column & grouped metrics
    train_group_df = _group_metrics_df(train_pred_df, epoch)

    # Save grouped CSV (train)
    train_group_df.to_csv(os.path.join(args.outdir, f'train_metrics_by_protein_epoch{epoch}.csv'), index=False)

    return l.avg, a.avg, u.avg, train_group_df


def validate(loader, model, crit, args, epoch):
    t = AverageMeter('Time', ':6.3f')
    l = AverageMeter('Loss', ':.4e')
    a = AverageMeter('Acc',  ':6.2f')
    u = AverageMeter('AUC',  ':6.2f')
    prog = ProgressMeter(len(loader), [t, l, a, u], prefix='Val: ')

    model.eval()

    all_logits = []
    all_labels = []
    all_barcodes = []

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(loader):
            inp = {k: v.cuda(non_blocking=True) for k, v in batch.items()
                   if k not in ['labels', 'barcodes']}
            lab = batch['labels'].cuda(non_blocking=True)
            barcodes = batch['barcodes']
            
            log = model(**inp)
            loss = crit(log, lab)

            acc, auc = compute_metrics(log.detach().cpu(), lab.detach().cpu())
            l.update(loss.item(), lab.size(0))
            a.update(acc, lab.size(0))
            u.update(auc, lab.size(0))

            all_logits.extend(log.cpu().numpy())
            all_labels.extend(lab.cpu().numpy())
            all_barcodes.extend(barcodes)

            t.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                prog.display(i)
        print(f"* Val loss {l.avg:.3f}")

    # per-sample preds (val)
    pred_df = pd.DataFrame({
        'barcode': all_barcodes,
        'true_label': all_labels,
        'pred_score': all_logits,
        'pred_label': [int(i >= 0.5) for i in all_logits]
    })
    pred_df.to_csv(os.path.join(args.outdir, f'val_preds_epoch{epoch}.csv'), index=False)

    # grouped metrics (val)
    group_df = _group_metrics_df(pred_df, epoch)
    group_df.to_csv(os.path.join(args.outdir, f'val_metrics_by_protein_epoch{epoch}.csv'), index=False)

    return l.avg, a.avg, u.avg, group_df

#-----------------------------------------------------------------------------
def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed); torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("CUDNN deterministic set; training may be slower.")
    main_worker(args.device, args)

if __name__ == '__main__':
    main()