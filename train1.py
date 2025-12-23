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
from imblearn.over_sampling import SMOTE

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
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if isinstance(inputs, dict):
            logits = inputs['logits']
        else:
            logits = inputs

        CE = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-CE)
        loss = self.alpha * (1 - pt)**self.gamma * CE

        return {'total_loss': loss.mean() if self.reduction == 'mean' else loss}


# ------------------------------
#   MODEL IMPORT
# ------------------------------

class MultiClassDeceptionDetector(nn.Module):
    def __init__(self, strategic_dim=256, text_dim=256, fusion_dim=512, n_classes=4, n_monte_carlo=10):
        super(MultiClassDeceptionDetector, self).__init__()
        
        from test import EmbeddingFusion, UncertaintyQuantification
        
        self.fusion = EmbeddingFusion(strategic_dim, text_dim, fusion_dim)
        self.uncertainty = UncertaintyQuantification(fusion_dim)
        self.n_monte_carlo = n_monte_carlo
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        
        self.consistency_head = nn.Linear(fusion_dim, 1)
        self.confidence_head = nn.Linear(fusion_dim, 1)
    
    def forward(self, strategic_emb, text_emb, training=True):
        fused_emb = self.fusion(strategic_emb, text_emb)
        epistemic, aleatoric = self.uncertainty(fused_emb)
        
        if training:
            logits = self.classifier(fused_emb)
            return {
                'logits': logits,
                'consistency_score': torch.sigmoid(self.consistency_head(fused_emb)),
                'confidence_score': torch.sigmoid(self.confidence_head(fused_emb)),
                'epistemic_uncertainty': epistemic,
                'aleatoric_uncertainty': aleatoric
            }
        else:
            self.train()
            preds = []
            for _ in range(self.n_monte_carlo):
                preds.append(F.softmax(self.classifier(fused_emb), dim=-1))
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


# ------------------------------
#   TRAINING LOOP
# ------------------------------

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, device, n_epochs=50, save_dir="./checkpoints"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    train_losses, val_losses, val_f1_scores = [], [], []
    best_val_f1 = 0.0
    
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
        
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)
        
        print(f"Epoch {epoch+1}: TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  ValF1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(save_dir, "best_model.pth"))
            print("New best model saved.")
        
        scheduler.step(val_loss)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1_scores': val_f1_scores,
        'best_val_f1': best_val_f1
    }

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
        'csv_path': 'data/final_dataset1.csv',
        'model_path': 'test_allminilm_finetuned-20250829T234732Z-1-001/test_allminilm_finetuned',
        'strategic_dim': 256,
        'text_dim': 256,
        'fusion_dim': 512,
        'batch_size': 32,
        'learning_rate': 2.76e-4,
        'n_epochs': 25,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': f'./deception_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print(config)

    # ---- Load data ----
    df = load_and_preprocess_data(config['csv_path'])

    # ---- Embeddings ----
    text_emb, strat_emb = generate_embeddings(df, config['model_path'])

    # ---- Labels ----
    labels, label_encoder = prepare_labels(df)
    n_classes = len(label_encoder.classes_)
    
    # 4. Split data
    X_text_train, X_text_temp, X_strat_train, X_strat_temp, y_train, y_temp = train_test_split(
        X_text, X_strat, y, test_size=0.3, random_state=42, stratify=y
    )

    X_text_val, X_text_test, X_strat_val, X_strat_test, y_val, y_test = train_test_split(
        X_text_temp, X_strat_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

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
    model = MultiClassDeceptionDetector(
        strategic_dim=config['strategic_dim'],
        text_dim=config['text_dim'],
        fusion_dim=config['fusion_dim'],
        n_classes=len(label_encoder.classes_)
    ).to(config['device'])
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # ---- Train ----
    history = train_model(
        train_loader, val_loader, model,
        criterion, optimizer, scheduler,
        config['device'], config['n_epochs'], config['save_dir']
    )
    
    # 9. Load best model and evaluate
    print("Loading best model for evaluation...")
    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluate_model(test_loader, model, config['device'], label_encoder)

    print(f"Training complete. Best Val F1 = {history['best_val_f1']:.4f}")


if __name__ == "__main__":
    main()
