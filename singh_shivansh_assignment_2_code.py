import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from torchvision.models import ResNet34_Weights
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "cifar10_data"
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# I uses these settings to save my GPU from dying
EPOCH_SLEEP  = 2      # seconds to sleep between epochs (gives GPU a breather)
ACCUM_STEPS  = 4      # gradient accumulation steps (effective batch = batch_size × accum)
                      # e.g. batch=64, accum=4 - effective batch of 256 but less GPU pressure
NUM_WORKERS = 2
PREFETCH    = 2
PIN_MEMORY  = True
PERSISTENT  = True

#Constant
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = 10
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram:.1f} GB")
    torch.backends.cudnn.benchmark    = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32   = True


#  Tranformers
#  All augmentation runs on CPU (torchvision transforms always run on CPU by default), so GPU only does the forward/backward pass.

MEAN = [0.4914, 0.4822, 0.4465] #Most commonly used values
STD  = [0.2470, 0.2435, 0.2616] #Same for this

cnn_train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
])
cnn_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

resnet_train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=8),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
])
resnet_test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

#Mixup
def mixup_batch(images, labels, alpha=0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#Data loading
def load_data():
    print("Loading official CIFAR-10 (auto-downloads if needed)")

    cnn_train_full = datasets.CIFAR10(DATA_DIR, train=True,  download=True,
                                      transform=cnn_train_transform)
    cnn_test       = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                      transform=cnn_test_transform)
    rn_train_full  = datasets.CIFAR10(DATA_DIR, train=True,  download=True,
                                      transform=resnet_train_transform)
    rn_test        = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                      transform=resnet_test_transform)

    def split(ds):
        n_tr = int(len(ds) * 0.9)
        n_va = len(ds) - n_tr
        return random_split(ds, [n_tr, n_va],
                            generator=torch.Generator().manual_seed(SEED))

    tr_cnn,    va_cnn    = split(cnn_train_full)
    tr_resnet, va_resnet = split(rn_train_full)

    print(f"CNN    -- train: {len(tr_cnn)}, val: {len(va_cnn)}, test: {len(cnn_test)}")
    print(f"ResNet -- train: {len(tr_resnet)}, val: {len(va_resnet)}, test: {len(rn_test)}")
    return tr_cnn, va_cnn, cnn_test, tr_resnet, va_resnet, rn_test


def make_loader(ds, batch_size, shuffle):
    return DataLoader(
        ds,
        batch_size         = batch_size,
        shuffle            = shuffle,
        num_workers        = NUM_WORKERS,
        pin_memory         = PIN_MEMORY,
        persistent_workers = PERSISTENT,
        prefetch_factor    = PREFETCH,
    )

#Model A Custom CNN
class CustomCNN(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — 64 filters
            nn.Conv2d(3,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),

            # Block 2 — 128 filters
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

            # Block 3 — 256 filters
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),

            # Block 4 — 512 filters
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout2d(0.4),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))

#Model 2 Fine tuned ResNet 34
def build_resnet(freeze_backbone=False, dropout=0.4):
    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


#  Training
# Uses gradient accumulation instead of one big batch hitting the GPU, does ACCUM_STEPS small batches and sum the gradients 

def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    scheduler=None, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup:
            imgs, labels_a, labels_b, lam = mixup_batch(imgs, labels)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            out = model(imgs)
            if use_mixup:
                loss = mixup_criterion(criterion, out, labels_a, labels_b, lam)
            else:
                loss = criterion(out, labels)
            # Scale loss by accum steps so gradients average correctly
            loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()

        # Only update weights every ACCUM_STEPS batches
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * ACCUM_STEPS * imgs.size(0)
        preds       = out.argmax(1)
        lbl         = labels_a if use_mixup else labels
        correct    += (preds == lbl).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            out  = model(imgs)
            loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds    = out.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train_model(model, train_loader, val_loader, cfg, label, use_mixup=True):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if cfg['optimizer'] == 'adam':
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg['lr'], weight_decay=cfg.get('wd', 1e-4))
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'],
                              momentum=0.9, weight_decay=cfg.get('wd', 1e-4),
                              nesterov=True)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = cfg['lr'],
        steps_per_epoch = len(train_loader) // ACCUM_STEPS,
        epochs          = cfg['epochs'],
        pct_start       = 0.3,
        anneal_strategy = 'cos',
    )

    scaler     = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    history    = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc   = 0.0
    no_improve = 0
    patience   = cfg.get('patience', 10)
    best_path  = OUT_DIR / f"best_{label}.pth"

    print(f"Training : {label}")
    print(f" Config   : {cfg}")
    print(f"Mixup    : {use_mixup}|Grad accum steps: {ACCUM_STEPS}")

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        tr_loss, tr_acc       = train_one_epoch(model, train_loader, optimizer,
                                                criterion, scaler,
                                                scheduler=scheduler,
                                                use_mixup=use_mixup)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        current_lr = optimizer.param_groups[0]['lr']
        elapsed    = time.time() - t0
        print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {va_loss:.4f}/{va_acc:.4f} | "
              f"LR {current_lr:.6f} | {elapsed:.1f}s")

        if va_acc > best_acc:
            best_acc   = va_acc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs")
                break
        if EPOCH_SLEEP > 0:
            torch.cuda.empty_cache()
            time.sleep(EPOCH_SLEEP)

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"Best val accuracy: {best_acc:.4f}")

    # Clear VRAM before next model
    torch.cuda.empty_cache()
    return history, best_acc

#Hyperparameter tuning
def hyperparameter_search(model_fn, train_ds, val_ds, param_grid,
                          model_label, use_mixup=True):
    results = []
    keys    = list(param_grid.keys())
    combos  = list(product(*param_grid.values()))

    print(f"HP search: {model_label}  ({len(combos)} combos)")

    for i, vals in enumerate(combos):
        cfg = dict(zip(keys, vals))
        cfg.setdefault('epochs', 15)
        cfg.setdefault('patience', 6)
        model     = model_fn(cfg).to(device)
        tr_loader = make_loader(train_ds, cfg['batch_size'], shuffle=True)
        va_loader = make_loader(val_ds,   cfg['batch_size'], shuffle=False)

        print(f"\n  Combo [{i+1}/{len(combos)}]: {cfg}")
        _, best_acc = train_model(model, tr_loader, va_loader, cfg,
                                  f"{model_label}_search_{i}", use_mixup=use_mixup)
        results.append({**cfg, 'val_acc': best_acc})

        # Clear VRAM between combos
        del model
        torch.cuda.empty_cache()

    results.sort(key=lambda x: x['val_acc'], reverse=True)
    print(f"\n  Results (best first):")
    for r in results:
        print(f"    {r}")
    return results

#Plot for comparison (Mostly for the report if I have space left)
def plot_history(history, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'],   label='Val',   linewidth=2)
    axes[0].set_title('Loss'); axes[0].legend()
    axes[0].set_xlabel('Epoch'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'],   label='Val',   linewidth=2)
    axes[1].set_title('Accuracy'); axes[1].legend()
    axes[1].set_xlabel('Epoch'); axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_confusion(labels, preds, title, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(cnn_val, cnn_test, rn_val, rn_test, save_path):
    x      = np.arange(2)
    width  = 0.3
    labels = ['Custom CNN', 'ResNet-34']
    val_accs  = [cnn_val,  rn_val]
    test_accs = [cnn_test, rn_test]

    plt.figure(figsize=(8, 5))
    b1 = plt.bar(x - width/2, val_accs,  width, label='Val',  color='steelblue', edgecolor='black')
    b2 = plt.bar(x + width/2, test_accs, width, label='Test', color='coral',     edgecolor='black')

    for bar, acc in list(zip(b1, val_accs)) + list(zip(b2, test_accs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{acc*100:.2f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    plt.xticks(x, labels); plt.ylim(0, 1.12)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison - Val vs Test Accuracy', fontsize=13, fontweight='bold')
    plt.legend(); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_hp_results(hp_results, title, save_path):
    df     = pd.DataFrame(hp_results).sort_values('val_acc', ascending=True)
    labels = [f"lr={r['lr']} bs={r['batch_size']} do={r['dropout']}"
              for _, r in df.iterrows()]
    plt.figure(figsize=(10, 5))
    plt.barh(labels, df['val_acc'].values, color='mediumseagreen', edgecolor='black')
    plt.xlabel('Val Accuracy'); plt.xlim(0, 1.0)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

#main
def main():
    # Load data
    tr_cnn, va_cnn, te_cnn, tr_resnet, va_resnet, te_resnet = load_data()

    # Hp search (Cnn)
    cnn_param_grid = {
        'lr':         [1e-3, 5e-4],
        'batch_size': [64, 128],
        'optimizer':  ['adam'],
        'dropout':    [0.3, 0.4],
        'wd':         [1e-4, 5e-4],
        'epochs':     [15],
    }
    def cnn_factory(cfg):
        return CustomCNN(dropout=cfg.get('dropout', 0.4))

    cnn_hp = hyperparameter_search(cnn_factory, tr_cnn, va_cnn,
                                   cnn_param_grid, "CNN", use_mixup=True)
    pd.DataFrame(cnn_hp).to_csv(OUT_DIR / "cnn_hp_results.csv", index=False)
    plot_hp_results(cnn_hp, "CNN Hyperparameter Search", OUT_DIR / "cnn_hp_search.png")
    # Cool down between model phases
    print("\nchill for some time")
    torch.cuda.empty_cache()
    time.sleep(30)

    # Hp search (resnet)
    resnet_param_grid = {
        'lr':         [5e-4, 1e-4],
        'batch_size': [64, 128],
        'optimizer':  ['adam'],
        'dropout':    [0.3, 0.4],
        'wd':         [1e-4, 5e-4],
        'epochs':     [15],
        'freeze':     [False],
    }
    def resnet_factory(cfg):
        return build_resnet(freeze_backbone=cfg.get('freeze', False),
                            dropout=cfg.get('dropout', 0.4))

    resnet_hp = hyperparameter_search(resnet_factory, tr_resnet, va_resnet,
                                      resnet_param_grid, "ResNet", use_mixup=True)
    pd.DataFrame(resnet_hp).to_csv(OUT_DIR / "resnet_hp_results.csv", index=False)
    plot_hp_results(resnet_hp, "ResNet Hyperparameter Search", OUT_DIR / "resnet_hp_search.png")

    # Full training
    best_cnn_cfg             = cnn_hp[0].copy()
    best_cnn_cfg['epochs']   = 80
    best_cnn_cfg['patience'] = 15

    best_rn_cfg              = resnet_hp[0].copy()
    best_rn_cfg['epochs']    = 60
    best_rn_cfg['patience']  = 12

    criterion = nn.CrossEntropyLoss()

    # CNN — final
    print("\nCNN final training")
    final_cnn  = CustomCNN(dropout=best_cnn_cfg.get('dropout', 0.4)).to(device)
    tr_ldr_cnn = make_loader(tr_cnn, best_cnn_cfg['batch_size'], shuffle=True)
    va_ldr_cnn = make_loader(va_cnn, 128, shuffle=False)
    te_ldr_cnn = make_loader(te_cnn, 128, shuffle=False)

    cnn_history, _ = train_model(final_cnn, tr_ldr_cnn, va_ldr_cnn,
                                  best_cnn_cfg, "final_cnn", use_mixup=True)
    _, cnn_val_acc,  cnn_val_preds,  cnn_val_labels  = evaluate(final_cnn, va_ldr_cnn, criterion)
    _, cnn_test_acc, cnn_test_preds, cnn_test_labels  = evaluate(final_cnn, te_ldr_cnn, criterion)

    print(f"\nCNN Val Accuracy  : {cnn_val_acc  * 100:.2f}%")
    print(f"CNN Test Accuracy : {cnn_test_acc * 100:.2f}%")
    print("\nTest Set Classification Report:")
    print(classification_report(cnn_test_labels, cnn_test_preds, target_names=CLASSES))

    # Cool down between models
    del final_cnn
    torch.cuda.empty_cache()
    print("\n[chill again")
    time.sleep(30)

    # ResNet — final
    print("\nFinal ResNet training")
    final_rn  = build_resnet(freeze_backbone=best_rn_cfg.get('freeze', False),
                             dropout=best_rn_cfg.get('dropout', 0.4)).to(device)
    tr_ldr_rn = make_loader(tr_resnet, best_rn_cfg['batch_size'], shuffle=True)
    va_ldr_rn = make_loader(va_resnet, 128, shuffle=False)
    te_ldr_rn = make_loader(te_resnet, 128, shuffle=False)

    rn_history, _ = train_model(final_rn, tr_ldr_rn, va_ldr_rn,
                                 best_rn_cfg, "final_resnet", use_mixup=True)
    _, rn_val_acc,  rn_val_preds,  rn_val_labels  = evaluate(final_rn, va_ldr_rn, criterion)
    _, rn_test_acc, rn_test_preds, rn_test_labels  = evaluate(final_rn, te_ldr_rn, criterion)

    print(f"\nResNet Val Accuracy  : {rn_val_acc  * 100:.2f}%")
    print(f"ResNet Test Accuracy : {rn_test_acc * 100:.2f}%")
    print("\nTest Set Classification Report:")
    print(classification_report(rn_test_labels, rn_test_preds, target_names=CLASSES))

    # Plot
    print("\n[plots")
    plot_history(cnn_history, "Custom CNN -- Training History",
                 OUT_DIR / "cnn_history.png")
    plot_history(rn_history,  "Fine-tuned ResNet-34 -- Training History",
                 OUT_DIR / "resnet_history.png")
    plot_confusion(cnn_test_labels, cnn_test_preds,
                   "CNN Confusion Matrix (Test)", OUT_DIR / "cnn_confusion.png")
    plot_confusion(rn_test_labels,  rn_test_preds,
                   "ResNet-34 Confusion Matrix (Test)", OUT_DIR / "resnet_confusion.png")
    plot_comparison(cnn_val_acc, cnn_test_acc, rn_val_acc, rn_test_acc,
                    OUT_DIR / "model_comparison.png")
    winner = "ResNet-34" if rn_test_acc > cnn_test_acc else "Custom CNN"
    diff   = abs(rn_test_acc - cnn_test_acc) * 100
    print(f"Custom CNN  -- Val: {cnn_val_acc*100:.2f}%  Test: {cnn_test_acc*100:.2f}%")
    print(f"ResNet-34   -- Val: {rn_val_acc*100:.2f}%  Test: {rn_test_acc*100:.2f}%")
    print(f"Winner      : {winner} (by {diff:.2f}% on test set)")
    print(f"CNN config  : {best_cnn_cfg}")
    print(f"ResNet cfg  : {best_rn_cfg}")
    print(f"Outputs     : {OUT_DIR}")

if __name__ == "__main__":
    main()