"""
Deception Detection with Cross-Attention Fusion
Implements attention-based fusion between text and strategic embeddings
to improve deception detection in Diplomacy game conversations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from sentence_transformers import SentenceTransformer
from gnn import get_strategic_embeddings
from imblearn.combine import SMOTETomek


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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


class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for fusing text and strategic embeddings"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # Query, Key, Value projections for cross-attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, query, key_value):
        """
        Args:
            query: (batch_size, dim) - queries from one modality
            key_value: (batch_size, dim) - keys and values from another modality
        Returns:
            (batch_size, dim) - attended features
        """
        batch_size = query.size(0)
        
        # Add sequence dimension for attention (batch, 1, dim)
        query = query.unsqueeze(1)
        key_value = key_value.unsqueeze(1)
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # (batch, 1, dim)
        K = self.k_proj(key_value)  # (batch, 1, dim)
        V = self.v_proj(key_value)  # (batch, 1, dim)
        
        # Reshape for multi-head attention: (batch, num_heads, 1, head_dim)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, 1, head_dim)
        
        # Reshape back: (batch, 1, dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.dim)
        
        # Output projection and residual connection
        output = self.out_proj(attn_output)
        output = output.squeeze(1)  # (batch, dim)
        
        # Residual connection and layer norm
        output = self.layer_norm(query.squeeze(1) + self.dropout(output))
        
        return output


class CrossAttentionDeceptionClassifier(nn.Module):
    """Attention-based fusion of text and strategic embeddings"""
    def __init__(self, strategic_dim=256, text_dim=256, n_classes=2, num_heads=4, dropout=0.3):
        super(CrossAttentionDeceptionClassifier, self).__init__()
        
        # Ensure both embeddings have the same dimension for attention
        self.strategic_proj = nn.Linear(strategic_dim, 256)
        self.text_proj = nn.Linear(text_dim, 256)
        
        # Cross-attention layers
        # Text attends to strategic information
        self.text_to_strategic_attn = CrossAttentionFusion(256, num_heads=num_heads, dropout=dropout)
        # Strategic attends to text information
        self.strategic_to_text_attn = CrossAttentionFusion(256, num_heads=num_heads, dropout=dropout)
        
        # Fusion layer to combine attended features
        self.fusion = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, strategic_emb, text_emb):
        # Project to common dimension
        strategic_features = self.strategic_proj(strategic_emb)
        text_features = self.text_proj(text_emb)
        
        # Cross-attention: each modality attends to the other
        text_attended = self.text_to_strategic_attn(text_features, strategic_features)
        strategic_attended = self.strategic_to_text_attn(strategic_features, text_features)
        
        # Combine attended features
        combined = torch.cat([text_attended, strategic_attended], dim=-1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


class SimpleDeceptionClassifier(nn.Module):
    """Much simpler architecture to reduce overfitting (Legacy - kept for compatibility)"""
    def __init__(self, strategic_dim=256, text_dim=256, n_classes=2, dropout=0.3):
        super(SimpleDeceptionClassifier, self).__init__()
        
        # Simple concatenation fusion
        input_dim = strategic_dim + text_dim
        
        # Smaller, simpler network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, n_classes)
        )
    
    def forward(self, strategic_emb, text_emb):
        # Simple concatenation
        x = torch.cat([strategic_emb, text_emb], dim=-1)
        return self.network(x)


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


def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df['sender_bool'] = df['sender_labels'].apply(parse_bool_label)
    df['receiver_bool'] = df['receiver_labels'].apply(parse_bool_label)
    df['deception_state'] = df.apply(
        lambda r: deception_state_from_bools(r['sender_bool'], r['receiver_bool']),
        axis=1
    )
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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in loader:
        strategic_emb = batch['strategic_embedding'].to(device)
        text_emb = batch['text_embedding'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(strategic_emb, text_emb)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, acc


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            strategic_emb = batch['strategic_embedding'].to(device)
            text_emb = batch['text_embedding'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(strategic_emb, text_emb)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, acc, all_preds, all_labels


def plot_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 1].plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_f1'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('F1 Score', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting Gap
    gap = [t - v for t, v in zip(history['train_loss'], history['val_loss'])]
    axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Overfitting Gap (Train - Val Loss)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"✓ Plot saved to {save_dir}/training_history.png")


def main():
    use_attention = True  # Set to True for CrossAttention, False for Simple model
    
    config = {
        'csv_path': 'data/final_dataset1.csv',
        'model_path': 'test_allminilm_finetuned-20250829T234732Z-1-001/test_allminilm_finetuned',
        'strategic_dim': 256,
        'text_dim': 256,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-3,  # Strong regularization
        'dropout': 0.4,
        'num_heads': 4,  # Number of attention heads for cross-attention
        'use_attention': use_attention,
        'n_epochs': 60,
        'patience': 20,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': f'./{"attention" if use_attention else "simple"}_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print("="*70)
    print("DECEPTION DETECTION MODEL")
    print("Cross-Attention Fusion" if config['use_attention'] else "Simple Concatenation")
    print("="*70)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    os.makedirs(config['save_dir'], exist_ok=True)

    # Load data
    df = load_and_preprocess_data(config['csv_path'])
    print(f"✓ Loaded {len(df)} samples")
    print(f"  Class distribution:\n{df['deception_state'].value_counts()}\n")

    # Generate embeddings
    print("Generating embeddings...")
    text_emb, strat_emb = generate_embeddings(df, config['model_path'])
    labels, label_encoder = prepare_labels(df)
    print(f"✓ Text embeddings: {text_emb.shape}")
    print(f"✓ Strategic embeddings: {strat_emb.shape}\n")

    print(f"✓ Strategic embeddings: {strat_emb.shape}\n")

    # -------------------------
    #  SMOTE + TOMEK
    # -------------------------
    print("Fusing embeddings for SMOTE+Tomek...")
    fused = np.concatenate([text_emb, strat_emb], axis=1)

    print("Applying SMOTETomek (this handles imbalance better than weights)...")
    # Using SMOTETomek to oversample minority AND clean overlapping majority
    smt = SMOTETomek(random_state=42)
    fused_resampled, labels_resampled = smt.fit_resample(fused, labels)
    
    # Un-fuse
    text_dim = text_emb.shape[1]
    X_text = fused_resampled[:, :text_dim]
    X_strat = fused_resampled[:, text_dim:]
    y = labels_resampled
    
    print(f"✓ Original size: {len(labels)}")
    print(f"✓ Resampled size: {len(y)}")
    
    # Split
    print("Splitting data (Train/Val/Test)...")
    X_text_train, X_text_temp, X_strat_train, X_strat_temp, y_train, y_temp = train_test_split(
        X_text, X_strat, y, test_size=0.3, random_state=42, stratify=y
    )

    X_text_val, X_text_test, X_strat_val, X_strat_test, y_val, y_test = train_test_split(
        X_text_temp, X_strat_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"✓ Train: {len(y_train)} samples")
    print(f"✓ Val:   {len(y_val)} samples")
    print(f"✓ Test:  {len(y_test)} samples\n")

    # Create datasets
    train_dataset = DeceptionDataset(X_text_train, X_strat_train, y_train)
    val_dataset = DeceptionDataset(X_text_val, X_strat_val, y_val)
    test_dataset = DeceptionDataset(X_text_test, X_strat_test, y_test)

    # Standard loaders (SMOTE balanced the data, so no weighted sampler needed)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Create model based on configuration
    if config['use_attention']:
        model = CrossAttentionDeceptionClassifier(
            strategic_dim=config['strategic_dim'],
            text_dim=config['text_dim'],
            n_classes=len(label_encoder.classes_),
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(config['device'])
        print(f"✓ Using CrossAttentionDeceptionClassifier with {config['num_heads']} attention heads")
    else:
        model = SimpleDeceptionClassifier(
            strategic_dim=config['strategic_dim'],
            text_dim=config['text_dim'],
            n_classes=len(label_encoder.classes_),
            dropout=config['dropout']
        ).to(config['device'])
        print("✓ Using SimpleDeceptionClassifier (concatenation-based)")

    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Focal Loss (better for hard examples)
    # Alpha can be tuned, reducing it slightly since SMOTE balanced the data
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['n_epochs'])

    # Training loop
    print("="*70)
    print("TRAINING")
    print("="*70)
    
    history = {
        'train_loss': [], 'train_f1': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_acc': []
    }
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(config['n_epochs']):
        # Train
        train_loss, train_f1, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )
        
        # Validate
        val_loss, val_f1, val_acc, _, _ = validate(
            model, val_loader, criterion, config['device']
        )
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        
        # Print
        print(f"Epoch {epoch+1:3d}/{config['n_epochs']} | "
              f"TrLoss: {train_loss:.4f} TrF1: {train_f1:.4f} TrAcc: {train_acc:.4f} | "
              f"VaLoss: {val_loss:.4f} VaF1: {val_f1:.4f} VaAcc: {val_acc:.4f}")
        
        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1
            }, os.path.join(config['save_dir'], "best_model.pth"))
            print(f"  ✓ New best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        scheduler.step()
    
    # Plot
    plot_history(history, config['save_dir'])

    # Test evaluation
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    checkpoint = torch.load(
        os.path.join(config['save_dir'], "best_model.pth"),
        map_location=config['device'],
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_f1, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, config['device']
    )
    
    print(f"\nTest F1:       {test_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(test_labels, test_preds, 
                                target_names=label_encoder.classes_, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    
    print("\nPer-Class Metrics:")
    for i, cls in enumerate(label_encoder.classes_):
        cls_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {cls:25s} - Accuracy: {cls_acc:.4f} ({cm[i, i]}/{cm[i].sum()})")
    
    print("\n" + "="*70)
    print(f"BEST VALIDATION F1: {best_val_f1:.4f}")
    print(f"FINAL TEST F1:      {test_f1:.4f}")
    print(f"Results saved to: {config['save_dir']}")
    print("="*70)


if __name__ == "__main__":
    main()