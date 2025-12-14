import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import copy

# Import custom modules
from network.TAM_model import MS_STANet_Pose
from Bullying10k import get_bullying10k_data


class Color:
    HEADER = '\033[95m';
    OKBLUE = '\033[94m';
    OKCYAN = '\033[96m';
    OKGREEN = '\033[92m'
    WARNING = '\033[93m';
    FAIL = '\033[91m';
    ENDC = '\033[0m';
    BOLD = '\033[1m';
    UNDERLINE = '\033[4m'


def plot_curve(y_true_bin, y_scores, num_classes, class_names, save_dir, epoch_name, curve_type='pr'):
    plt.figure(figsize=(12, 9))
    class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    all_aucs = []

    for i in range(num_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            if curve_type == 'pr':
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
                area = auc(recall, precision)
                plt.plot(recall, precision, lw=2, label=f'PR Class {class_names[i]} (AUC = {area:.2f})')
            else:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                area = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'ROC Class {class_names[i]} (AUC = {area:.2f})')
            all_aucs.append(area)

    macro_auc = np.nanmean(all_aucs) if all_aucs else 0.0
    if curve_type == 'roc': plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall' if curve_type == 'pr' else 'False Positive Rate')
    plt.ylabel('Precision' if curve_type == 'pr' else 'True Positive Rate')
    plt.title(f'Multi-class {curve_type.upper()} Curve ({epoch_name})\nMacro Avg AUC: {macro_auc:.2f}')
    plt.legend(loc="best");
    plt.grid(True)

    curve_path = os.path.join(save_dir, f'{curve_type}_curve_{epoch_name}.png')
    try:
        plt.savefig(curve_path)
    except Exception as e:
        logging.warning(f"{Color.WARNING}Failed to save {curve_type.upper()} curve: {e}{Color.ENDC}")
    plt.close()
    return macro_auc


def get_optimizer_and_scheduler(model, args):
    optimizer_class = {'sgd': optim.SGD, 'adam': optim.Adam, 'adamw': optim.AdamW}.get(args.opt.lower())
    optimizer_kwargs = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.opt.lower() == 'sgd': optimizer_kwargs['momentum'] = 0.9
    
    # Standard Optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs,
                                                   eta_min=args.lr * 0.01)
    elif args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    return optimizer, scheduler


def _current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_model(args):
    # --- Logging Config ---
    # Rename run to indicate it uses Aux Loss
    run_name = f"{args.model_name}_Aux_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"{Color.HEADER}{'=' * 60}\n{Color.BOLD}{Color.OKCYAN}{f'🚀 Initializing Training (Baseline + AuxLoss) 🚀':^68}\n{'=' * 60}{Color.ENDC}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "evaluation_curves"), exist_ok=True)
    
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard_logs"))
    writer.add_text("Hyperparameters", pd.DataFrame(vars(args).items()).to_markdown())
    
    logging.info(f"Device: {Color.BOLD}{device}{Color.ENDC}")
    
    # Set seed
    try:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    logging.info(f"Outputs will be saved to: {Color.OKGREEN}{run_dir}{Color.ENDC}")

    logging.info(f"\n{Color.BOLD}{Color.OKCYAN}📊 Loading Dataset...{Color.ENDC}")
    train_loader, val_loader, test_loader, num_classes, class_names = get_bullying10k_data(
        data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
        step=args.step, gap=args.gap, size=args.size, use_train_augmentations=args.use_data_augmentation
    )

    logging.info(f"\n{Color.BOLD}{Color.OKCYAN}🧠 Building Model & Optimizer...{Color.ENDC}")
    model = MS_STANet_Pose(num_classes=num_classes, dropout=args.dropout).to(device)
    logging.info(
        f"Total Parameters: {Color.BOLD}{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M{Color.ENDC}")
    
    # Standard Cross Entropy Loss
    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)
    
    logging.info(
        f"\n{Color.HEADER}{'=' * 60}\n{Color.BOLD}{Color.OKCYAN}{f'✨ Starting Training ({args.epochs} Epochs) ✨':^68}\n{'=' * 60}{Color.ENDC}")
    logging.info(f"Configuration: Aux Loss Weight = {args.aux_loss_weight}")

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        logging.info(
            f"\n{Color.BOLD}Epoch {epoch + 1}/{args.epochs} | LR: {_current_lr(optimizer):.2e}{Color.ENDC}")

        if epoch < args.warmup_epochs:
            lr_scale = (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * lr_scale

        # --- Training Loop ---
        model.train()
        running_loss, running_corrects, total_samples = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"{Color.OKGREEN}Epoch {epoch + 1} [Train]{Color.ENDC}", ncols=120,
                          file=sys.stdout)

        for events, labels, poses in train_pbar:
            events, labels, poses = events.permute(0, 2, 1, 3, 4).to(device), labels.to(device), poses.to(device)
            optimizer.zero_grad()
            
            # Forward with features for Aux Loss
            outputs, evt_aux, pose_aux = model(events, poses, return_features=True)
            
            # Calculate Losses
            main_loss = criterion(outputs, labels)
            
            # Aux loss (average of event and pose aux heads)
            if args.aux_loss_weight > 0:
                aux_loss = 0.5 * (criterion(evt_aux, labels) + criterion(pose_aux, labels))
                loss = main_loss + args.aux_loss_weight * aux_loss
            else:
                loss = main_loss
                
            loss.backward()
            optimizer.step()

            preds = torch.max(outputs, 1)[1]
            running_loss += loss.item() * events.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += events.size(0)
            train_pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('LearningRate', _current_lr(optimizer), epoch)
        logging.info(f"  └── {Color.OKGREEN}[Train Completed]{Color.ENDC} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        # --- Validation Loop ---
        if val_loader:
            model.eval()
            val_loss, val_corrects, val_total = 0.0, 0, 0
            with torch.no_grad():
                for events, labels, poses in tqdm(val_loader,
                                                  desc=f"{Color.OKBLUE}Epoch {epoch + 1} [Valid]{Color.ENDC}", ncols=120,
                                                  file=sys.stdout):
                    events, labels, poses = events.permute(0, 2, 1, 3, 4).to(device), labels.to(device), poses.to(device)
                    # No need for features in validation
                    outputs = model(events, poses)
                    
                    loss = criterion(outputs, labels)
                    preds = torch.max(outputs, 1)[1]
                    val_loss += loss.item() * events.size(0)
                    val_corrects += torch.sum(preds == labels.data)
                    val_total += events.size(0)

            epoch_loss_val = val_loss / val_total
            epoch_acc_val = val_corrects.double() / val_total
            writer.add_scalar('Loss/val', epoch_loss_val, epoch)
            writer.add_scalar('Accuracy/val', epoch_acc_val, epoch)
            logging.info(
                f"  └── {Color.OKBLUE}[Valid Completed]{Color.ENDC} Loss: {epoch_loss_val:.4f} | Acc: {epoch_acc_val:.4f}")

            if epoch_acc_val > best_val_acc:
                best_val_acc = epoch_acc_val
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(run_dir, "models", 'best_model.pth'))
                logging.info(
                    f"      └── 🎉 {Color.OKGREEN}New Best! Val Acc: {best_val_acc:.4f}. Model saved.{Color.ENDC}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                logging.info(f"{Color.WARNING}Early Stopping triggered: No improvement for {args.patience} epochs.{Color.ENDC}")
                break

        if scheduler and epoch >= args.warmup_epochs:
            scheduler.step()

    # --- Final Test ---
    logging.info(f"\n{Color.HEADER}{'=' * 60}\n{Color.BOLD}{Color.OKCYAN}{'🧪 Starting Final Test 🧪':^68}\n{'=' * 60}{Color.ENDC}")
    
    ckpt_cli = getattr(args, 'checkpoint_path', '')
    best_model_path = ckpt_cli if (isinstance(ckpt_cli, str) and os.path.isfile(ckpt_cli)) \
        else os.path.join(run_dir, "models", 'best_model.pth')
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logging.info(f"Loaded weights for testing: {best_model_path}")
    else:
        logging.warning(f"{Color.WARNING}Warning: Best model not found. Using current model for testing.{Color.ENDC}")

    model.eval()
    test_loss, test_corrects, test_total = 0.0, 0, 0
    all_probs_test, all_labels_test = [], []
        
    with torch.no_grad():
        for events, labels, poses in tqdm(test_loader, desc=f"{Color.OKCYAN}Testing{Color.ENDC}", ncols=120,
                                          file=sys.stdout):
            events, labels, poses = events.permute(0, 2, 1, 3, 4).to(device), labels.to(device), poses.to(device)
            logits = model(events, poses)
            
            loss = criterion(logits, labels)
            preds = torch.max(logits, 1)[1]
            test_loss += loss.item() * events.size(0)
            test_corrects += torch.sum(preds == labels.data)
            test_total += events.size(0)
            all_probs_test.append(F.softmax(logits, dim=1).cpu())
            all_labels_test.append(labels.cpu())

    epoch_loss_test = test_loss / test_total
    epoch_acc_test = test_corrects.double() / test_total

    y_true_bin = label_binarize(torch.cat(all_labels_test).numpy(), classes=list(range(num_classes)))
    y_scores = torch.cat(all_probs_test).numpy()
    pr_auc = plot_curve(y_true_bin, y_scores, num_classes, class_names, os.path.join(run_dir, "evaluation_curves"),
                        "final_test", 'pr')
    roc_auc = plot_curve(y_true_bin, y_scores, num_classes, class_names, os.path.join(run_dir, "evaluation_curves"),
                         "final_test", 'roc')

    logging.info(
        f"\n{Color.HEADER}{'=' * 60}\n{Color.BOLD}{Color.OKGREEN}{'🎉 All Process Completed! 🎉':^68}\n{'=' * 60}{Color.ENDC}")
    logging.info(f"Final Test Accuracy: {Color.BOLD}{epoch_acc_test:.4f}{Color.ENDC}")
    logging.info(f"Final Test Macro PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual-Modal (Event+Pose) Training Script')
    parser.add_argument('--data_root', default=r'./dataset', type=str, help='Root directory of the dataset')
    parser.add_argument('--model_name', default='ms_stanet_pose', type=str, help='Model name')
    parser.add_argument('--output_dir', default=r'./runs_output', type=str, help='Directory to save outputs')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', type=str)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--step', default=16, type=int)
    parser.add_argument('--gap', default=4, type=int)
    parser.add_argument('--size', default=112, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--num_workers', default=8, type=int, help='Set to 0 on Windows')
    parser.add_argument('--use_data_augmentation', action='store_true', default=True, help='Basic Data Augmentation')
    parser.add_argument('--patience', default=30, type=int)
    
    # === Auxiliary Loss ===
    parser.add_argument('--aux_loss_weight', default=0.2, type=float, help='Weight for auxiliary loss (0 to disable)')
    
    # Testing & Misc
    parser.add_argument('--test_only', action='store_true', default=False, help='Run testing only')
    parser.add_argument('--checkpoint_path', default='', type=str, help='Path to checkpoint for testing')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')

    args = parser.parse_args()
    
    if args.test_only:
        if args.checkpoint_path:
            import os
            if os.path.isfile(args.checkpoint_path):
                args.output_dir = os.path.dirname(os.path.dirname(args.checkpoint_path))
        args.epochs = 0
        train_model(args)
    else:
        train_model(args)