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

# Import your existing modules
from sentence_transformers import SentenceTransformer
from gnn import get_strategic_embeddings

# Your existing model classes (assuming they're in the same file or imported)
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


def parse_bool_label(v):
    """
    Robust mapping to True / False / None.
    Accepts booleans, ints, and common strings like 'True','False','NOANNOTATION'.
    Returns: True | False | None
    """
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
    """Map sender/receiver boolean labels to deception states"""
    
    # now both are booleans
    if sender_b and (not receiver_b):
        return 'no_deception'   # sender said true, receiver didn't => unsuccessful suspicion
    if sender_b and receiver_b:
        return 'no_deception'             # both true => no deception / no suspicion
    if (not sender_b) and receiver_b:
        return 'successful_deception'     # sender false, receiver true => deception succeeded
    if (not sender_b) and (not receiver_b):
        return 'successful_deception'   # both false => unsuccessful deception
   


# class MultiClassDeceptionLoss(nn.Module):
#     def __init__(self, class_weights=None, alpha=1.0, beta=0.5, gamma=0.3):
#         super(MultiClassDeceptionLoss, self).__init__()
#         self.alpha = alpha  # Main classification loss weight
#         self.beta = beta    # Consistency loss weight
#         self.gamma = gamma  # Confidence loss weight
#         self.class_weights = class_weights
        
#     def forward(self, predictions, targets):
#         # Main cross-entropy loss
#         if self.class_weights is not None:
#             ce_loss = F.cross_entropy(predictions['logits'], targets, weight=self.class_weights)
#         else:
#             ce_loss = F.cross_entropy(predictions['logits'], targets)
        
#         # Get probabilities for auxiliary losses
#         probs = F.softmax(predictions['logits'], dim=-1)
#         max_probs = torch.max(probs, dim=-1)[0]
        
#         # Consistency regularization (should be high when prediction is confident)
#         consistency_loss = F.mse_loss(
#             predictions['consistency_score'].squeeze(),
#             max_probs
#         )
        
#         # Confidence calibration loss
#         pred_classes = torch.argmax(probs, dim=-1)
#         confidence_target = (pred_classes == targets).float()
#         confidence_loss = F.mse_loss(predictions['confidence_score'].squeeze(), confidence_target)
        
#         # Uncertainty regularization
#         uncertainty_loss = predictions['aleatoric_uncertainty'].mean()
        
#         total_loss = (self.alpha * ce_loss + 
#                      self.beta * consistency_loss + 
#                      self.gamma * confidence_loss + 
#                      0.1 * uncertainty_loss)
        
#         return {
#             'total_loss': total_loss,
#             'ce_loss': ce_loss,
#             'consistency_loss': consistency_loss,
#             'confidence_loss': confidence_loss,
#             'uncertainty_loss': uncertainty_loss
#         }
class FocalLoss(nn.Module):
    def __init__(self,alpha,gamma,reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.reduction=reduction

    def forward(self,inputs,targets):
        if isinstance(inputs, dict):
            logits = inputs['logits']
        else:
            logits = inputs
        BCE_loss=nn.functional.cross_entropy(logits,targets,reduction='none')
        pt=torch.exp(-BCE_loss)
        F_loss=self.alpha*(1-pt)**self.gamma*BCE_loss
        if self.reduction=='mean':
            loss_value = torch.mean(F_loss)
        elif self.reduction=='sum':
            loss_value = torch.sum(F_loss)
        else:
            loss_value = F_loss
        return {'total_loss': loss_value}
class MultiClassDeceptionDetector(nn.Module):
    def __init__(self, strategic_dim=256, text_dim=256, fusion_dim=512, n_classes=4, n_monte_carlo=10):
        super(MultiClassDeceptionDetector, self).__init__()
        
        from test import EmbeddingFusion, UncertaintyQuantification
        
        self.fusion = EmbeddingFusion(strategic_dim, text_dim, fusion_dim)
        self.uncertainty = UncertaintyQuantification(fusion_dim)
        self.n_monte_carlo = n_monte_carlo
        self.n_classes = n_classes
        
        # Main classifier - modified for multi-class
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)  # Changed from 1 to n_classes
        )
        
        # Auxiliary heads for multi-task learning
        self.consistency_head = nn.Linear(fusion_dim, 1)
        self.confidence_head = nn.Linear(fusion_dim, 1)
    
    def forward(self, strategic_emb, text_emb, training=True):
        # Fusion
        fused_emb = self.fusion(strategic_emb, text_emb)
        
        # Get uncertainty estimates
        epistemic, aleatoric = self.uncertainty(fused_emb)
        
        if training:
            # Single forward pass during training
            logits = self.classifier(fused_emb)
            consistency_score = torch.sigmoid(self.consistency_head(fused_emb))
            confidence_score = torch.sigmoid(self.confidence_head(fused_emb))
            
            return {
                'logits': logits,
                'consistency_score': consistency_score,
                'confidence_score': confidence_score,
                'epistemic_uncertainty': epistemic,
                'aleatoric_uncertainty': aleatoric
            }
        else:
            # Monte Carlo dropout for uncertainty
            self.train()  # Enable dropout
            predictions = []
            
            for _ in range(self.n_monte_carlo):
                logits = self.classifier(fused_emb)
                predictions.append(F.softmax(logits, dim=-1))
            
            self.eval()
            predictions = torch.stack(predictions)
            
            # Calculate uncertainty metrics
            mean_pred = predictions.mean(dim=0)
            epistemic_uncertainty = predictions.var(dim=0).mean(dim=-1)
            
            consistency_score = torch.sigmoid(self.consistency_head(fused_emb))
            confidence_score = torch.sigmoid(self.confidence_head(fused_emb))
            
            return {
                'probabilities': mean_pred,
                'epistemic_uncertainty': epistemic_uncertainty,
                'aleatoric_uncertainty': aleatoric.squeeze(),
                'consistency_score': consistency_score,
                'confidence_score': confidence_score,
                'prediction_variance': epistemic_uncertainty
            }

def load_and_preprocess_data(csv_path):
    """Load and preprocess the training data"""
    print("Loading and preprocessing data...")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Apply boolean label parsing
    df['sender_bool'] = df['sender_labels'].apply(parse_bool_label)
    df['receiver_bool'] = df['receiver_labels'].apply(parse_bool_label)
    
    # Generate deception states
    df['deception_state'] = df.apply(
        lambda r: deception_state_from_bools(r['sender_bool'], r['receiver_bool']), 
        axis=1
    )
    
    print("Deception state distribution:")
    print(df['deception_state'].value_counts(dropna=False))
    
    # Remove unknown states or handle them separately
    df_clean = df[df['deception_state'] != 'unknown'].copy()
    print(f"Data after removing unknowns: {len(df_clean)} samples")
    
    return df_clean

def generate_embeddings(df, model_path):
    """Generate text and strategic embeddings for all samples"""
    print("Generating embeddings...")
    
    # Load fine-tuned text embedding model
    text_model = SentenceTransformer(model_path)
    
    # Generate text embeddings
    print("Generating text embeddings...")
    messages = df['messages'].tolist()
    text_embeddings = []
    
    for msg in tqdm(messages, desc="Text embeddings"):
        emb = text_model.encode(msg, normalize_embeddings=True)
        text_embeddings.append(emb)
    
    text_embeddings = np.array(text_embeddings)
    
    # Generate strategic embeddings
    print("Generating strategic embeddings...")
    strategic_embeddings = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Strategic embeddings"):
        # Prepare game context
        game_context = {
            'game_score': row.get('game_score', 0),
            'game_score_delta': row.get('game_score_delta', 0),
            'absolute_message_index': row.get('absolute_message_index', 0),
            'relative_message_index': row.get('relative_message_index', 0)
        }
        
        # Get strategic embeddings from GNN
        embeddings = get_strategic_embeddings(
            row['speakers'], 
            row['receivers'], 
            row['messages'], 
            game_context
        )
        
        # Use sender embedding
        strategic_embeddings.append(embeddings[0])
    
    strategic_embeddings = np.array(strategic_embeddings)
    
    return text_embeddings, strategic_embeddings

def prepare_labels(df):
    """Prepare and encode labels"""
    # Create label encoder
    label_encoder = LabelEncoder()
    
    # Encode labels
    labels = label_encoder.fit_transform(df['deception_state'])
    
    print("Label mapping:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{i}: {class_name}")
    
    return labels, label_encoder

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, device, n_epochs=50, save_dir="./checkpoints"):
    """Training loop"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    val_f1_scores = []
    best_val_f1 = 0.0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_preds = []
        train_targets = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training"):
            strategic_emb = batch['strategic_embedding'].to(device)
            text_emb = batch['text_embedding'].to(device)
            labels = batch['label'].view(-1).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(strategic_emb, text_emb, training=True)
            losses = criterion(outputs, labels)
            
            # Backward pass
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += losses['total_loss'].item()
            
            # Collect predictions for metrics
            preds = torch.argmax(outputs['logits'], dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Validation"):
                strategic_emb = batch['strategic_embedding'].to(device)
                text_emb = batch['text_embedding'].to(device)
                labels = batch['label'].view(-1).to(device)
                
                outputs = model(strategic_emb, text_emb, training=True)
                losses = criterion(outputs, labels)
                
                total_val_loss += losses['total_loss'].item()
                
                preds = torch.argmax(outputs['logits'], dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)
        
        print(f"Epoch {epoch+1}/{n_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  New best model saved! F1: {val_f1:.4f}")
        
        # Step scheduler
        if scheduler:
            scheduler.step(val_loss)
        
        print("-" * 50)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1_scores': val_f1_scores,
        'best_val_f1': best_val_f1
    }

def evaluate_model(test_loader, model, device, label_encoder):
    """Evaluate model on test set"""
    model.eval()
    test_preds = []
    test_targets = []
    uncertainties = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            strategic_emb = batch['strategic_embedding'].to(device)
            text_emb = batch['text_embedding'].to(device)
            labels = batch['label'].view(-1).to(device)
            
            outputs = model(strategic_emb, text_emb, training=False)
            
            preds = torch.argmax(outputs['probabilities'], dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            uncertainties.extend(outputs['epistemic_uncertainty'].cpu().numpy())
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, 
                              target_names=label_encoder.classes_))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_targets, test_preds))
    
    return test_preds, test_targets, uncertainties

def main():
    """Main training pipeline"""
    
    # Configuration
    config = {
        'csv_path': 'data/final_dataset1.csv',  # Your training CSV path
        'model_path': 'test_allminilm_finetuned-20250829T234732Z-1-001/test_allminilm_finetuned',
        'strategic_dim': 256,
        'text_dim': 256,  # Should match your fine-tuned model's output dim
        'fusion_dim': 512,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'n_epochs': 25,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': f'./deception_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print(f"Using device: {config['device']}")
    print(f"Configuration: {config}")
    
    # 1. Load and preprocess data
    df = load_and_preprocess_data(config['csv_path'])
    
    # 2. Generate embeddings
    text_embeddings, strategic_embeddings = generate_embeddings(df, config['model_path'])
    
    # 3. Prepare labels
    labels, label_encoder = prepare_labels(df)
    n_classes = len(label_encoder.classes_)
    
    # 4. Split data
    X_text_train, X_text_temp, X_strat_train, X_strat_temp, y_train, y_temp = train_test_split(
        text_embeddings, strategic_embeddings, labels, 
        test_size=0.3, random_state=42, stratify=labels
    )
    
    X_text_val, X_text_test, X_strat_val, X_strat_test, y_val, y_test = train_test_split(
        X_text_temp, X_strat_temp, y_temp, 
        test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_text_train)}, Val: {len(X_text_val)}, Test: {len(X_text_test)}")
    
    # 5. Create datasets and dataloaders
    train_dataset = DeceptionDataset(X_text_train, X_strat_train, y_train)
    val_dataset = DeceptionDataset(X_text_val, X_strat_val, y_val)
    test_dataset = DeceptionDataset(X_text_test, X_strat_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 6. Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights).to(config['device'])
    
    # 7. Initialize model, loss, and optimizer
    model = MultiClassDeceptionDetector(
        strategic_dim=config['strategic_dim'],
        text_dim=config['text_dim'],
        fusion_dim=config['fusion_dim'],
        n_classes=n_classes
    ).to(config['device'])
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 8. Train model
    print("Starting training...")
    training_history = train_model(
        train_loader, val_loader, model, criterion, optimizer, scheduler,
        config['device'], config['n_epochs'], config['save_dir']
    )
    
    # 9. Load best model and evaluate
    print("Loading best model for evaluation...")
    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_preds, test_targets, uncertainties = evaluate_model(test_loader, model, config['device'], label_encoder)
    
    # 10. Save everything needed for inference
    print("Saving model and metadata...")
    
    # Save label encoder
    with open(os.path.join(config['save_dir'], 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save config
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        config_serializable = config.copy()
        config_serializable['device'] = str(config['device'])
        json.dump(config_serializable, f, indent=2)
    
    # Save training history
    with open(os.path.join(config['save_dir'], 'training_history.pkl'), 'wb') as f:
        pickle.dump(training_history, f)
    
    print(f"Training complete! Best validation F1: {training_history['best_val_f1']:.4f}")
    print(f"Model and artifacts saved to: {config['save_dir']}")
    
    return model, label_encoder, config

if __name__ == "__main__":
    model, label_encoder, config = main()