import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import json
from datetime import datetime
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE  # Needed for SMOTETomek parameter

# Import your existing modules
from sentence_transformers import SentenceTransformer
from gnn import get_strategic_embeddings


# ------------------------------
#  DATASET + SAMPLER
# ------------------------------

class DeceptionDataset(Dataset):
    def __init__(self, text_embeddings, strategic_embeddings, labels):
        self.text_embeddings = text_embeddings
        self.strategic_embeddings = strategic_embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
        'text_embedding': torch.FloatTensor(self.text_embeddings[idx]),
        'strategic_embedding': torch.FloatTensor(self.strategic_embeddings[idx]),
        'label': torch.tensor(self.labels[idx], dtype=torch.long)
    }


# ------------------------------
#   LABEL PROCESSING HELPERS
# ------------------------------

def parse_bool_label(v):
    if pd.isna(v):
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {'true', 't', '1', 'yes', 'y'}:
        return True
    if s in {'false', 'f', '0', 'no', 'n'}:
        return False
    return None


def deception_state_from_bools(sender_b, receiver_b):
    if sender_b and (not receiver_b):
        return 'no_deception'
    if sender_b and receiver_b:
        return 'no_deception'
    if (not sender_b) and receiver_b:
        return 'successful_deception'
    if (not sender_b) and (not receiver_b):
        return 'successful_deception'


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        if isinstance(inputs, dict):
            logits = inputs['logits']
        else:
            logits = inputs

        # Apply label smoothing
        if self.label_smoothing > 0:
            n_classes = logits.size(-1)
            # Smooth labels
            smoothed_targets = torch.zeros_like(logits)
            smoothed_targets.fill_(self.label_smoothing / (n_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # Compute cross entropy with smooth labels
            log_probs = F.log_softmax(logits, dim=-1)
            CE = -(smoothed_targets * log_probs).sum(dim=-1)
        else:
            CE = F.cross_entropy(logits, targets, reduction='none')
        
        pt = torch.exp(-CE)
        loss = self.alpha * (1 - pt)**self.gamma * CE

        return {'total_loss': loss.mean() if self.reduction == 'mean' else loss}


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


def mixup_data(x1, x2, y, alpha=0.3):
    """Apply mixup augmentation to embeddings"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x1, mixed_x2, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup"""
    loss_a = criterion(pred, y_a)['total_loss']
    loss_b = criterion(pred, y_b)['total_loss']
    return {'total_loss': lam * loss_a + (1 - lam) * loss_b}



# ------------------------------
#   MODEL IMPORT
# ------------------------------

class MultiClassDeceptionDetector(nn.Module):
    def __init__(self, strategic_dim=256, text_dim=256, fusion_dim=512, n_classes=4, n_monte_carlo=10, 
                 dropout1=0.4, dropout2=0.3, num_heads=16, attention_dropout=0.1):
        super(MultiClassDeceptionDetector, self).__init__()
        
        from test import EmbeddingFusion, UncertaintyQuantification
        
        # Enhanced fusion with more attention heads and dropout
        self.fusion = EmbeddingFusion(
            strategic_dim, 
            text_dim, 
            fusion_dim,
            num_heads=num_heads,  # Configurable attention heads
            attention_dropout=attention_dropout  # Attention-specific dropout
        )
        self.uncertainty = UncertaintyQuantification(fusion_dim)
        self.n_monte_carlo = n_monte_carlo
        
        # DEEPER classification head: 512 → 256 → 128 → 64 → n_classes
        self.classifier = nn.Sequential(
            # Layer 1: 512 → 256
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout1),
            
            # Layer 2: 512 → 256
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout2),
            
            # Layer 3: 256 → 128
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout2),
            
            # Layer 4: 128 → 64
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout2 * 0.5),  # Reduced dropout near output
            
            # Layer 5: 64 → n_classes
            nn.Linear(64, n_classes)
        )
        
        # Residual projection from fusion to classification
        self.residual_classifier = nn.Linear(fusion_dim, n_classes)
        
        self.consistency_head = nn.Linear(fusion_dim, 1)
        self.confidence_head = nn.Linear(fusion_dim, 1)
    
    def forward(self, strategic_emb, text_emb, training=True):
        fused_emb = self.fusion(strategic_emb, text_emb)
        epistemic, aleatoric = self.uncertainty(fused_emb)
        
        if training:
            # Main classification path
            logits = self.classifier(fused_emb)
            
            # Residual skip connection
            residual_logits = self.residual_classifier(fused_emb)
            final_logits = logits + residual_logits  # Combine paths
            
            return {
                'logits': final_logits,
                'consistency_score': torch.sigmoid(self.consistency_head(fused_emb)),
                'confidence_score': torch.sigmoid(self.confidence_head(fused_emb)),
                'epistemic_uncertainty': epistemic,
                'aleatoric_uncertainty': aleatoric
            }
        else:
            self.train()
            preds = []
            for _ in range(self.n_monte_carlo):
                logits = self.classifier(fused_emb)
                residual_logits = self.residual_classifier(fused_emb)
                final_logits = logits + residual_logits
                preds.append(F.softmax(final_logits, dim=-1))
            self.eval()
            preds = torch.stack(preds)
            
            mean_pred = preds.mean(dim=0)
            epistemic_unc = preds.var(dim=0).mean(dim=-1)
            
            return {
                'probabilities': mean_pred,
                'epistemic_uncertainty': epistemic_unc,
                'aleatoric_uncertainty': aleatoric.squeeze(),
            }


# ------------------------------
#   DATA PROCESSING
# ------------------------------

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # processed_data.csv already has 'deception_state' column
    # Just filter out any missing values
    df = df[df['deception_state'].notna()]
    
    return df


def generate_embeddings(df, model_path):
    text_model = SentenceTransformer(model_path)
    messages = df['messages'].tolist()

    text_embeddings = text_model.encode(messages, normalize_embeddings=True, show_progress_bar=True)
    strategic_embeddings = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        game_context = {
            'game_score': row.get('game_score', 0),
            'game_score_delta': row.get('game_score_delta', 0),
            'absolute_message_index': row.get('absolute_message_index', 0),
            'relative_message_index': row.get('relative_message_index', 0)
        }

        strategic_embeddings.append(
            get_strategic_embeddings(
                row['speakers'],
                row['receivers'],
                row['messages'],
                game_context
            )[0]
        )
    
    return np.array(text_embeddings), np.array(strategic_embeddings)


def prepare_labels(df):
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df['deception_state'])
    return labels, encoder


# ------------------------------
#   ENHANCED TRAINING LOOP
# ------------------------------

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, device, 
                config, swa_model=None, swa_scheduler=None, save_dir="./checkpoints"):
    """
    Enhanced training loop with:
    - Mixup augmentation
    - Warmup + Cosine LR scheduling
    - Stochastic Weight Averaging (SWA)
    - Multi-metric early stopping
    - Comprehensive metric tracking
    """
    os.makedirs(save_dir, exist_ok=True)
    
    train_losses, val_losses, val_f1_scores = [], [], []
    train_f1_scores, train_accs, val_accs = [], [], []
    learning_rates = []
    
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    use_mixup = config.get('use_mixup', False)
    mixup_alpha = config.get('mixup_alpha', 0.3)
    use_swa = config.get('use_swa', False) and swa_model is not None
    swa_start = config.get('swa_start_epoch', 30)
    patience = config.get('patience', 10)
    min_delta = config.get('min_delta', 0.0001)
    n_epochs = config.get('n_epochs', 50)
    
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Mixup: {use_mixup} | SWA: {use_swa} | Patience: {patience}")
    print(f"{'='*70}\n")
    
    for epoch in range(n_epochs):
        # ---- Training ----
        model.train()
        total_train_loss = 0
        train_preds, train_targets = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training"):
            strategic_emb = batch['strategic_embedding'].to(device)
            text_emb = batch['text_embedding'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Apply mixup augmentation
            if use_mixup and np.random.random() < 0.5:  # 50% probability
                mixed_s, mixed_t, y_a, y_b, lam = mixup_data(strategic_emb, text_emb, labels, mixup_alpha)
                outputs = model(mixed_s, mixed_t, training=True)
                losses = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(strategic_emb, text_emb, training=True)
                losses = criterion(outputs, labels)

            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += losses['total_loss'].item()
            train_preds.extend(torch.argmax(outputs['logits'], dim=-1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        # ---- Validation ----
        model.eval()
        total_val_loss = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                strategic_emb = batch['strategic_embedding'].to(device)
                text_emb = batch['text_embedding'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(strategic_emb, text_emb, training=True)
                losses = criterion(outputs, labels)
                
                total_val_loss += losses['total_loss'].item()
                val_preds.extend(torch.argmax(outputs['logits'], dim=-1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        train_acc = (np.array(train_preds) == np.array(train_targets)).mean()
        val_acc = (np.array(val_preds) == np.array(val_targets)).mean()
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update learning rate
        current_lr = scheduler.step()
        learning_rates.append(current_lr)
        
        # SWA update
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            if swa_scheduler is not None:
                swa_scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{n_epochs} | " +
              f"TrLoss: {train_loss:.4f} TrF1: {train_f1:.4f} TrAcc: {train_acc:.4f} | " +
              f"VaLoss: {val_loss:.4f} VaF1: {val_f1:.4f} VaAcc: {val_acc:.4f} | " +
              f"LR: {current_lr:.6f}")
        
        # Multi-metric early stopping (F1 improvement OR loss improvement)
        f1_improved = val_f1 > (best_val_f1 + min_delta)
        loss_improved = val_loss < (best_val_loss - min_delta)
        
        if f1_improved or loss_improved:
            improvement_msg = []
            if f1_improved:
                best_val_f1 = val_f1
                improvement_msg.append(f"F1: {val_f1:.4f}")
            if loss_improved:
                best_val_loss = val_loss
                improvement_msg.append(f"Loss: {val_loss:.4f}")
            
            patience_counter = 0
            
            # Save best model
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss
            }
            torch.save(save_dict, os.path.join(save_dir, "best_model.pth"))
            print(f"  ✓ New best model saved ({', '.join(improvement_msg)})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n{'='*70}")
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"{'='*70}\n")
            break
    
    # Final SWA update
    if use_swa and swa_model is not None:
        print(f"\n{'='*70}")
        print("Finalizing SWA model...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save({
            'model_state_dict': swa_model.module.state_dict(),
        }, os.path.join(save_dir, "swa_model.pth"))
        print(f"✓ SWA model saved")
        print(f"{'='*70}\n")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1_scores': train_f1_scores,
        'val_f1_scores': val_f1_scores,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'learning_rates': learning_rates,
        'best_val_f1': best_val_f1,
        'best_val_loss': best_val_loss
    }


def plot_training_history(history, save_dir):
    """Plot enhanced training metrics and save to file"""
    epochs = range(1, len(history['train_losses']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Losses
    axes[0, 0].plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: F1 Scores
    axes[0, 1].plot(epochs, history['train_f1_scores'], 'b-', label='Training F1', linewidth=2)
    axes[0, 1].plot(epochs, history['val_f1_scores'], 'g-', label='Validation F1', linewidth=2)
    axes[0, 1].set_title('F1 Score', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Accuracy
    axes[0, 2].plot(epochs, history['train_accs'], 'b-', label='Training Acc', linewidth=2)
    axes[0, 2].plot(epochs, history['val_accs'], 'orange', label='Validation Acc', linewidth=2)
    axes[0, 2].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epochs')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    axes[1, 0].plot(epochs, history['learning_rates'], 'purple', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Overfitting Gap
    gap = [t - v for t, v in zip(history['train_losses'], history['val_losses'])]
    axes[1, 1].plot(epochs, gap, 'red', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Overfitting Gap (Train - Val Loss)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: F1 Comparison
    axes[1, 2].bar(['Train', 'Val'], 
                   [history['train_f1_scores'][-1], history['val_f1_scores'][-1]],
                   color=['blue', 'green'], alpha=0.7)
    axes[1, 2].set_title('Final F1 Scores', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('F1 Score')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"✓ Training history plot saved to {save_dir}/training_history.png")


# ------------------------------
#   EVALUATION
# ------------------------------

def evaluate_swa_model(test_loader, swa_model, device, label_encoder):
    """Evaluate SWA (Stochastic Weight Averaging) model"""
    print("\n" + "="*70)
    print("SWA MODEL EVALUATION")
    print("="*70)
    
    swa_model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            strategic_emb = batch['strategic_embedding'].to(device)
            text_emb = batch['text_embedding'].to(device)
            labels = batch['label'].to(device)
            
            output = swa_model(strategic_emb, text_emb, training=False)
            pred = torch.argmax(output['probabilities'], dim=-1)
            
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='weighted')
    
    print(f"\n✓ SWA Test Accuracy: {acc:.4f}")
    print(f"✓ SWA Test F1: {f1:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(trues, preds, target_names=label_encoder.classes_))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(trues, preds)
    print(cm)
    
    print("\nPer-Class Metrics:")
    for i, cls in enumerate(label_encoder.classes_):
        cls_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {cls:25s} - Accuracy: {cls_acc:.4f} ({cm[i, i]}/{cm[i].sum()})")
    
    return acc, f1


def evaluate_model(test_loader, model, device, label_encoder):
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            strategic_emb = batch['strategic_embedding'].to(device)
            text_emb = batch['text_embedding'].to(device)
            labels = batch['label'].to(device)
            
            output = model(strategic_emb, text_emb, training=False)
            pred = torch.argmax(output['probabilities'], dim=-1)
            
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    
    print("\nClassification Report:")
    print(classification_report(trues, preds, target_names=label_encoder.classes_))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(trues, preds))


# ------------------------------
#   MAIN TRAINING PIPELINE
# ------------------------------

def main():
    config = {
        'csv_path': 'data/processed_data.csv',
        'model_path': 'test_allminilm_finetuned-20250829T234732Z-1-001/test_allminilm_finetuned',
        'strategic_dim': 256,
        'text_dim': 256,
        'fusion_dim': 512,
        'batch_size': 32,
        'learning_rate': 1.532033526528867e-4,
        'n_epochs': 25,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': f'./deception_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        # Enhanced regularization params
        'smote_k_neighbors': 3,
        'label_smoothing': 0.1,
        'use_mixup': True,
        'mixup_alpha': 0.3,
        'dropout1': 0.22798466779344834,
        'dropout2': 0.293627659799008,
        
        # Training improvements
        'warmup_epochs': 5,
        'use_swa': True,
        'swa_start_epoch': 20,  # 60% of 25 epochs
        'patience': 10,
        'min_delta': 0.0001,
        
        # Architecture enhancements
        'num_heads': 16,  # Attention heads (increased from 8)
        'attention_dropout': 0.1,  # Attention-specific dropout
        
        # Loss params
        'focal_alpha': 0.78530692087020266,
        'focal_gamma': 2.740172718188119,
        'weight_decay': 1.1615339145351501e-05
    }
    
    print("="*70)
    print("ENHANCED DECEPTION DETECTION MODEL")
    print("="*70)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*70)

    # ---- Load data ----
    df = load_and_preprocess_data(config['csv_path'])

    # ---- Embeddings ----
    text_emb, strat_emb = generate_embeddings(df, config['model_path'])

    # ---- Labels ----
    labels, label_encoder = prepare_labels(df)

    # ---- Train / Val / Test Split (BEFORE SMOTE to prevent leakage) ----
    print("\n" + "="*70)
    print("SPLITTING DATA (Before SMOTE to prevent leakage)")
    print("="*70)
    print(f"Original class distribution: {np.bincount(labels)}")
    
    # First split: separate test set (this will NOT be resampled)
    X_text_trainval, X_text_test, X_strat_trainval, X_strat_test, y_trainval, y_test = train_test_split(
        text_emb, strat_emb, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    # Second split: separate validation set (this will NOT be resampled)
    X_text_train, X_text_val, X_strat_train, X_strat_val, y_train, y_val = train_test_split(
        X_text_trainval, X_strat_trainval, y_trainval, test_size=0.176, random_state=42, stratify=y_trainval
    )  # 0.176 * 0.85 ≈ 0.15, so we get 70/15/15 split
    
    print(f"Train set: {len(y_train)} samples - {np.bincount(y_train)}")
    print(f"Val set:   {len(y_val)} samples - {np.bincount(y_val)}")
    print(f"Test set:  {len(y_test)} samples - {np.bincount(y_test)}")
    print("="*70 + "\n")

    # -------------------------
    #  SMOTE + TOMEK (FIXED - Applied ONLY to training data)
    # -------------------------
    print("="*70)
    print("APPLYING SMOTE+TOMEK (Only to training data)")
    print("="*70)
    
    # Fuse training embeddings
    fused_train = np.concatenate([X_text_train, X_strat_train], axis=1)
    
    print(f"Training class distribution before SMOTE: {np.bincount(y_train)}")
    print(f"Applying SMOTETomek (k_neighbors={config.get('smote_k_neighbors', 5)})...")
    
    # Apply SMOTETomek ONLY to training data
    smt = SMOTETomek(
        smote=SMOTE(k_neighbors=config.get('smote_k_neighbors', 5), random_state=42, sampling_strategy=1.0),
        random_state=42
    )
    fused_train_resampled, y_train_resampled = smt.fit_resample(fused_train, y_train)
    
    print(f"Training class distribution after SMOTE: {np.bincount(y_train_resampled)}")
    print("="*70 + "\n")

    # Un-fuse the resampled training data
    text_dim = X_text_train.shape[1]
    X_text_train = fused_train_resampled[:, :text_dim]
    X_strat_train = fused_train_resampled[:, text_dim:]
    y_train = y_train_resampled
    
    # Validation and test sets remain unchanged (no SMOTE applied)

    # ---- Dataloaders ----
    train_dataset = DeceptionDataset(X_text_train, X_strat_train, y_train)
    val_dataset   = DeceptionDataset(X_text_val, X_strat_val, y_val)
    test_dataset  = DeceptionDataset(X_text_test, X_strat_test, y_test)

    # Weighted sampler for balanced training
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = class_weights[y_train]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
    )

    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # ---- Model ----
    print(f"\n{'='*70}")
    print("INITIALIZING MODEL")
    print(f"{'='*70}")
    
    model = MultiClassDeceptionDetector(
        strategic_dim=config['strategic_dim'],
        text_dim=config['text_dim'],
        fusion_dim=config['fusion_dim'],
        n_classes=len(label_encoder.classes_),
        dropout1=config['dropout1'],
        dropout2=config['dropout2'],
        num_heads=config.get('num_heads', 16),  # Enhanced attention
        attention_dropout=config.get('attention_dropout', 0.1)  # Attention dropout
    ).to(config['device'])
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Architecture: {config.get('num_heads', 16)} attention heads, deeper 5-layer classifier")

    # Enhanced FocalLoss with label smoothing
    criterion = FocalLoss(
        alpha=config['focal_alpha'],
        gamma=config['focal_gamma'],
        label_smoothing=config['label_smoothing']
    )
    print(f"✓ FocalLoss (alpha={config['focal_alpha']:.3f}, gamma={config['focal_gamma']:.3f}, " +
          f"smoothing={config['label_smoothing']})")
    
    # Optimizer with weight decay
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Warmup Cosine Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['n_epochs']
    )
    print(f"✓ WarmupCosineScheduler (warmup={config['warmup_epochs']} epochs)")
    
    # SWA (Stochastic Weight Averaging)
    swa_model = None
    if config['use_swa']:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=config['learning_rate'] * 0.1)
        print(f"✓ SWA enabled (start_epoch={config['swa_start_epoch']})")
    
    print(f"{'='*70}\n")

    # ---- Train ----
    history = train_model(
        train_loader, val_loader, model,
        criterion, optimizer, scheduler,
        config['device'], config,
        swa_model=swa_model if config['use_swa'] else None,
        swa_scheduler=swa_scheduler if config['use_swa'] else None,
        save_dir=config['save_dir']
    )

    # Plot metrics
    plot_training_history(history, config['save_dir'])

    # ---- Evaluate ----
    print("Loading best model...")
    checkpoint = torch.load(
        os.path.join(config['save_dir'], "best_model.pth"),
        map_location=config['device'],
        weights_only=False  # PyTorch 2.6 compatibility
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluate_model(test_loader, model, config['device'], label_encoder)

    # ---- Evaluate SWA Model (if available) ----
    if config['use_swa'] and os.path.exists(os.path.join(config['save_dir'], "swa_model.pth")):
        print("\n" + "="*70)
        print("Loading SWA model for comparison...")
        print("="*70)
        
        swa_checkpoint = torch.load(
            os.path.join(config['save_dir'], "swa_model.pth"),
            map_location=config['device'],
            weights_only=False
        )
        
        # Create a new model instance for SWA
        swa_model = MultiClassDeceptionDetector(
            strategic_dim=config['strategic_dim'],
            text_dim=config['text_dim'],
            fusion_dim=config['fusion_dim'],
            n_classes=len(label_encoder.classes_),
            dropout1=config['dropout1'],
            dropout2=config['dropout2'],
            num_heads=config.get('num_heads', 16),
            attention_dropout=config.get('attention_dropout', 0.1)
        ).to(config['device'])
        
        swa_model.load_state_dict(swa_checkpoint['model_state_dict'])
        
        swa_acc, swa_f1 = evaluate_swa_model(test_loader, swa_model, config['device'], label_encoder)
        
        # Compare results
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(f"Best Checkpoint - Val F1: {history['best_val_f1']:.4f}")
        print(f"SWA Model       - Test F1: {swa_f1:.4f}")
        if swa_f1 > history['best_val_f1']:
            print(f"✓ SWA model is BETTER by {(swa_f1 - history['best_val_f1'])*100:.2f}%!")
        else:
            print(f"Best checkpoint is better by {(history['best_val_f1'] - swa_f1)*100:.2f}%")
        print("="*70)

    print(f"\nTraining complete. Best Val F1 = {history['best_val_f1']:.4f}")


if __name__ == "__main__":
    main()