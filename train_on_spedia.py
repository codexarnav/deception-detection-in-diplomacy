import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sentence_transformers import SentenceTransformer

from test import EmbeddingFusion, UncertaintyQuantification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report as sk_classification_report
import copy
from torch_geometric.nn import GATv2Conv

model_path = 'test_allminilm_finetuned-20250829T234732Z-1-001/test_allminilm_finetuned'

class SpediaDataset(Dataset):
    def __init__(self, textual_embeddings, labels):
        self.textual_embeddings = textual_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_emb = torch.FloatTensor(self.textual_embeddings[idx])
        return {
            'text_embedding': text_emb,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class StrategicGNN(nn.Module):
    """Simplified Strategic GNN"""
    
    def __init__(self, node_features: int, message_dim: int, embedding_dim: int = 256):
        super(StrategicGNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Message encoder
        self.message_encoder = nn.Sequential(
            nn.Linear(message_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        # Graph convolutions
        combined_dim = 128  # 64 + 64
        self.conv1 = GATv2Conv(combined_dim, 32, heads=2, concat=True, dropout=0.0)
        self.conv2 = GATv2Conv(64, 32, heads=2, concat=True, dropout=0.0)
        self.conv3 = GATv2Conv(64, embedding_dim, heads=1, concat=False, dropout=0.0)
        # Enhancement layer
        self.enhancer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, node_features, message_features, edge_index):
        # Encode inputs
        msg_emb = self.message_encoder(message_features)
        node_emb = self.node_encoder(node_features)
        
        # Combine
        x = torch.cat([node_emb, msg_emb], dim=1)
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        
        embeddings = self.conv3(x, edge_index)
        
        # Enhance
        strategic_embeddings = self.enhancer(embeddings)
        
        return strategic_embeddings


def load_and_preprocess_data(csv_path):
    data = pd.read_csv(csv_path)
    # Drop rows with missing values in required columns
    required_cols = ['Agent_name', 'User', 'Full_log', 'Description']
    # Ensure Description is numeric and only keep 0 and 1
    data['Description'] = pd.to_numeric(data['Description'], errors='coerce')
    data = data[data['Description'].isin([0, 1])].copy()
    data = data.reset_index(drop=True)
    print(f"Unique values in Description after filtering: {data['Description'].unique()}")
    return data

# Utility to remove 'comm' field from Full_log
def remove_comm_from_log(log):
    import re
    # Remove comm="..." or comm='...'
    log = re.sub(r'comm=\".*?\"', '', log)
    log = re.sub(r"comm='.*?'", '', log)
    return log
def remove_name_from_log(log):
    import re
    # Remove comm="..." or comm='...'
    log = re.sub(r'name=\".*?\"', '', log)
    log = re.sub(r"name='.*?'", '', log)
    return log
def remove_a0_from_log(log):
    import re
    # Remove comm="..." or comm='...'
    log = re.sub(r'a0=\".*?\"', '', log)
    log = re.sub(r"a0='.*?'", '', log)
    return log

# Model class definition moved out of function scope

# Simple classifier for textual embeddings only
class TextOnlyClassifier(nn.Module):
    def __init__(self, text_dim=384, n_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, text_emb):
        logits = self.classifier(text_emb)
        return logits

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

def generate_textual_embeddings(messages, model_path):
    text_model = SentenceTransformer(model_path)
    textual_embeddings = text_model.encode(messages, normalize_embeddings=True, show_progress_bar=True)
    return textual_embeddings

def generate_strategic_embeddings(data):
    from tqdm import tqdm
    senders = data['Agent_name'].tolist()
    receivers = data['User'].tolist()
    messages = data['Full_log'].tolist()
    n_samples = len(senders)
    strategic_embeddings = []
    print("Generating strategic embeddings with progress bar...")

    # --- Hyperparameters (should match your GNN) ---
    node_features = 25
    message_dim = 384  # MiniLM embedding size (must match MiniLM output)
    embedding_dim = 256

    # Initialize GNN model ONCE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = StrategicGNN(node_features, message_dim, embedding_dim).to(device)
    gnn_model.eval()

    # Dummy scaler for node features (replace with real scaler if needed)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Fit scaler on dummy data (or real node features if available)
    scaler.fit(np.random.randn(10, node_features))


    def build_features(row):
        # Modular feature extraction from dataset row
        # Example: [is_sender, is_receiver, command, level, activity, action, ...]
        # You can expand this as needed
        features_sender = np.zeros(node_features)
        features_receiver = np.zeros(node_features)

        # Example: one-hot for sender/receiver
        features_sender[0] = 1.0
        features_receiver[1] = 1.0

        def safe_float(val):
            try:
                return float(val)
            except:
                return 0.0

        

        # Pad or truncate to node_features
        features_sender = features_sender[:node_features]
        features_receiver = features_receiver[:node_features]
        return np.stack([features_sender, features_receiver])

    # Cache the text model for efficiency
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    def get_message_embedding(message):
        emb = text_model.encode([message], normalize_embeddings=True)
        return np.repeat(emb, 2, axis=0)

    # Build edge index (bidirectional)
    edge_index = torch.LongTensor([[0, 1], [1, 0]]).t().contiguous()

    for idx, row in enumerate(tqdm(data.to_dict(orient='records'), total=n_samples)):
        sender = row['Agent_name']
        receiver = row['User']
        message = row['Full_log']
        node_features_arr = build_features(row)
        node_features_arr = scaler.transform(node_features_arr)
        msg_emb_arr = get_message_embedding(message)

        node_tensor = torch.FloatTensor(node_features_arr).to(device)
        msg_tensor = torch.FloatTensor(msg_emb_arr).to(device)
        edge_idx = edge_index.to(device)

        with torch.no_grad():
            emb = gnn_model(node_tensor, msg_tensor, edge_idx)
        strategic_embeddings.append(emb[0].cpu().numpy())
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {n_samples} strategic embeddings...")
    return np.array(strategic_embeddings)



def train_pipeline(csv_path, model_path, batch_size=32, epochs=10, lr=1e-4):
    """
    Training pipeline with optional hard split and augmentation
    
    Args:
        use_hard_split: If True, split by agent (Solution 1)
        use_augmentation: If True, add noise to embeddings (Solution 3)
        noise_level: Standard deviation of Gaussian noise (0.1-0.2 recommended)
    """
    data = load_and_preprocess_data(csv_path)
    print(f"Loaded {len(data)} rows after filtering and cleaning.")
    
    # Always use random stratified split for balanced train/val
    print("Using RANDOM stratified split (recommended)...")
    # Remove 'comm' from Full_log before embedding
    data['Full_log'] = data['Full_log'].apply(remove_comm_from_log)
    data['Full_log']=data['Full_log'].apply(remove_name_from_log)
    data['Full_log']=data['Full_log'].apply(remove_a0_from_log)
    textual_embeddings = generate_textual_embeddings(data['Full_log'].tolist(), model_path)
    labels = data['Description'].values
    indices = np.arange(len(labels))
    X_train, X_val, y_train, y_val, text_train, text_val = train_test_split(
        indices, labels, textual_embeddings, 
        test_size=0.2, random_state=42
    )

    print(f"\n[Class Distribution]")
    print(f"Train: Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
    print(f"Val: Class 0: {(y_val==0).sum()}, Class 1: {(y_val==1).sum()}")    

    train_dataset = SpediaDataset(text_train, y_train)
    val_dataset = SpediaDataset(text_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = TextOnlyClassifier(text_dim=textual_embeddings.shape[1], n_classes=2)
    model = model.cuda() if torch.cuda.is_available() else model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    val_f1_scores = []
    
    print("\n[Starting Training]")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # Print sample embeddings and labels for inspection
    print("\nSample textual embedding:", textual_embeddings[0][:5])
    print("Sample label:", labels[0])

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            text_emb = batch['text_embedding']
            labels_batch = batch['label']

            if torch.cuda.is_available():
                text_emb = text_emb.cuda()
                labels_batch = labels_batch.cuda()

            logits = model(text_emb)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                text_emb = batch['text_embedding']
                labels_batch = batch['label']

                if torch.cuda.is_available():
                    text_emb = text_emb.cuda()
                    labels_batch = labels_batch.cuda()

                logits = model(text_emb)
                preds_probs = F.softmax(logits, dim=-1)
                preds = preds_probs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_batch.cpu().numpy())

                loss = criterion(logits, labels_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate F1 score
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_f1_scores.append(val_f1)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
        target_names = ['deception', 'no-deception']
        print(classification_report(all_labels, all_preds, digits=4, target_names=target_names))
    
    # Plot learning curves
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(val_f1_scores, label='Val F1 Score', marker='d', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves_hard_split.png', dpi=150)
    plt.close()
    print("\nLearning curves saved as 'learning_curves_hard_split.png'")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    
    print(f"{'='*80}\n")


def main():
    csv_path = 'filtered_data.csv'
    
   # Noise std (0.1-0.2 recommended)
    
    print("="*80)
    print("CONFIGURATION")
    print("="*80)
    
    print("="*80 + "\n")
    
    train_pipeline(
        csv_path, 
        model_path,
        batch_size=32,
        epochs=10,  # Increased since model should be harder to overfit
        lr=1e-4,
        
    )

if __name__ == "__main__":
    main()