#!/usr/bin/env python3

#modification of train_reformer_bc.py that can use different DNABERT-style encoders (foundation models)
#defaults to plant-DNABERT-BPE
#binary classification

import os, random, time, math, warnings, argparse, shutil, h5py
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, roc_auc_score
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
        return enc, label

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
    auc  = roc_auc_score(labels.cpu(), logits.cpu())
    return acc, auc

#-----------------------------------------------------------------------------
#Collate
def build_collate(tokenizer):
    def collate(batch):
        features, labels = zip(*batch)
        features = tokenizer.pad(features, return_tensors='pt')
        #drop token_type_ids so the wrapper never sees it
        features.pop('token_type_ids', None)
        labels = torch.stack(labels)
        features['labels'] = labels
        return features
    return collate


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
    # base.config.to_json_file(os.path.join(args.outdir, "config.json"))  # <-- 移到下面

    # encoder -----------------------------------------------------------------
    base = T.AutoModel.from_pretrained(args.model_name,
                                       trust_remote_code=True)
    base.resize_token_embeddings(len(tok))
    model = Bert4BinaryClassification(base)

    base.config.to_json_file(os.path.join(args.outdir, "config.json"))  # <-- 放到这里

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
        validate(val_ld, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_epoch(trn_ld, model, criterion, scheduler, optim, epoch, args)
        auc = validate(val_ld, model, criterion, args)
        is_best = auc > best_auc
        best_auc = max(auc, best_auc)
        torch.save(model.module.state_dict(),
                   f"{args.outdir}/model_epoch{epoch}.bin")
        if is_best:
            torch.save(model.module.state_dict(),
                       f"{args.outdir}/model_best.bin")

#-----------------------------------------------------------------------------
#Train & validate loops
def train_epoch(loader, model, crit, sched, optim, epoch, args):
    t = AverageMeter('Time', ':6.3f')
    d = AverageMeter('Data', ':6.3f')
    l = AverageMeter('Loss', ':.4e')
    a = AverageMeter('Acc',  ':6.2f')
    u = AverageMeter('AUC',  ':6.2f')
    prog = ProgressMeter(len(loader), [t, d, l, a, u],
                         prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    for i, batch in enumerate(loader):
        d.update(time.time() - end)
        inp = {k: v.cuda(non_blocking=True) for k, v in batch.items()
               if k != 'labels'}
        lab = batch['labels'].cuda(non_blocking=True)

        log = model(**inp)
        loss = crit(log, lab)

        acc, auc = compute_metrics(log.detach().cpu(), lab.detach().cpu())
        l.update(loss.item(), lab.size(0))
        a.update(acc, lab.size(0))
        u.update(auc, lab.size(0))

        optim.zero_grad()
        loss.backward()
        optim.step()
        sched.step()

        t.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            prog.display(i)

def validate(loader, model, crit, args):
    t = AverageMeter('Time', ':6.3f')
    l = AverageMeter('Loss', ':.4e')
    a = AverageMeter('Acc',  ':6.2f')
    u = AverageMeter('AUC',  ':6.2f')
    prog = ProgressMeter(len(loader), [t, l, a, u], prefix='Val: ')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(loader):
            inp = {k: v.cuda(non_blocking=True) for k, v in batch.items()
                   if k != 'labels'}
            lab = batch['labels'].cuda(non_blocking=True)
            log = model(**inp)
            loss = crit(log, lab)

            acc, auc = compute_metrics(log.detach().cpu(), lab.detach().cpu())
            l.update(loss.item(), lab.size(0))
            a.update(acc, lab.size(0))
            u.update(auc, lab.size(0))

            t.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                prog.display(i)
        print(f"* Val loss {l.avg:.3f}")

    return u.avg

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
