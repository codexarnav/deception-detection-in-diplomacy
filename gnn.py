import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleTextProcessor:
    """Simplified text processor with key strategic patterns"""
    
    def __init__(self):
        self.patterns = {
            'strategic': r'\b(?:attack|defend|alliance|treaty|war|peace|negotiate|support|help)\b',
            'deception': r'\b(?:maybe|perhaps|possibly|might|could|uncertain|honestly|frankly)\b',
            'urgency': r'\b(?:urgent|immediate|quickly|asap|now|hurry|critical)\b',
            'cooperation': r'\b(?:together|alliance|partner|cooperate|collaborate|mutual)\b',
            'aggression': r'\b(?:attack|destroy|eliminate|war|fight|battle|hostile)\b'
        }
    
    def extract_features(self, text: str) -> dict:
        """Extract 8 key text features"""
        text = str(text).lower()
        words = text.split()
        
        features = {}
        
        # Basic counts (3 features)
        features['word_count'] = len(words)
        features['sentence_count'] = max(1, len(re.findall(r'[.!?]+', text)))
        features['exclamation_count'] = text.count('!')
        
        # Strategic patterns (5 features)
        for pattern_name, pattern in self.patterns.items():
            matches = len(re.findall(pattern, text))
            features[f'{pattern_name}_score'] = matches / max(len(words), 1)
        
        return features

class StrategicGNN(nn.Module):
    """Simplified Strategic GNN"""
    
    def __init__(self, node_features: int, message_dim: int, embedding_dim: int = 256):
        super(StrategicGNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Message encoder
        self.message_encoder = nn.Sequential(
            nn.Linear(message_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )
        
        # Graph convolutions
        combined_dim = 256  # 128 + 128
        self.conv1 = GATv2Conv(combined_dim, 64, heads=4, concat=True, dropout=0.2)
        self.conv2 = GATv2Conv(256, 64, heads=2, concat=True, dropout=0.2)
        self.conv3 = GATv2Conv(128, embedding_dim, heads=1, concat=False, dropout=0.1)
        
        # Enhancement layer
        self.enhancer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim),
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

# Global instances
_text_processor = SimpleTextProcessor()
_gnn_model = None
_message_encoder = None
_scaler = StandardScaler()
_is_initialized = False

def _initialize_models():
    """Initialize models once"""
    global _gnn_model, _message_encoder, _scaler, _is_initialized
    
    if _is_initialized:
        return
    
    # Initialize message encoder  
    _message_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    _message_encoder.to(device)
    
    # Initialize scaler with dummy data (20 features after removing deception patterns)
    dummy_features = np.random.randn(10, 20)
    _scaler.fit(dummy_features)
    
    _is_initialized = True

def get_strategic_embeddings(sender: str, receiver: str, message: str, 
                           game_context: Optional[Dict] = None) -> np.ndarray:
    """
    Generate strategic embeddings for sender-receiver pair
    Returns embeddings for both sender and receiver based on this interaction
    """
    
    # Initialize models
    _initialize_models()
    
    # Default context
    if game_context is None:
        game_context = {
            'game_score': 100,
            'game_score_delta': 0,
            'absolute_message_index': 1,
            'relative_message_index': 1
        }
    
    # Extract text features using same logic
    text_features = _text_processor.extract_features(message)
    
    # Encode message using MiniLM
    clean_msg = str(message).strip()
    if len(clean_msg) < 3:
        clean_msg = "empty message"
    
    message_emb = _message_encoder.encode(
        f"Diplomatic message: {clean_msg}",
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Build node features for sender and receiver (exactly 25 features each)
    countries = [sender, receiver]
    node_features = []
    node_message_embeddings = []
    
    for i, country in enumerate(countries):
        features = []
        
        # 1. Performance metrics (5 features) - same as original
        score = game_context.get('game_score', 100)
        score_delta = game_context.get('game_score_delta', 0)
        
        features.extend([
            float(score),  # game_score mean
            float(abs(score_delta)),  # game_score std (approximation)
            float(score_delta),  # game_score_delta mean
            float(abs(score_delta) * 0.5),  # game_score_delta std
            float(0.5)  # performance_rank (default middle)
        ])
        
        # 2. Activity metrics (4 features) - same as original  
        if country == sender:
            # Sender has higher activity
            features.extend([
                0.5,  # Activity ratio
                1.0,  # Sender ratio (this is sender)
                0.0,  # Receiver ratio
                1.0   # Send/receive ratio
            ])
        else:
            # Receiver has lower activity
            features.extend([
                0.5,  # Activity ratio
                0.0,  # Sender ratio
                1.0,  # Receiver ratio (this is receiver)  
                0.0   # Send/receive ratio
            ])
        
        # 3. Text features (8 features) - from message content
        text_cols = ['word_count', 'sentence_count', 'exclamation_count', 
                   'strategic_score', 'deception_score', 'urgency_score', 
                   'cooperation_score', 'aggression_score']
        for col in text_cols:
            features.append(float(text_features.get(col, 0.0)))
        
        # 4. Temporal features (3 features) - message timing
        features.extend([
            float(game_context.get('absolute_message_index', 1)),
            float(game_context.get('relative_message_index', 1)),
            float(1)  # message_position default
        ])
        
        # Total: 5 (performance) + 4 (activity) + 8 (text) + 3 (temporal) = 20 features
        # Removed deception pattern features to prevent potential target leakage
        
        # Ensure exactly 20 features
        features = features[:20]
        while len(features) < 20:
            features.append(0.0)
        
        node_features.append(features)
        node_message_embeddings.append(message_emb)  # Same message embedding for both
    
    # Convert to arrays and scale - same as original
    node_features = np.array(node_features)
    node_features = _scaler.transform(node_features)
    node_message_embeddings = np.array(node_message_embeddings)
    
    # Build simple edge - sender to receiver and back
    edges = [(0, 1), (1, 0)]  # Bidirectional edge between sender(0) and receiver(1)
    edge_index = torch.LongTensor(edges).t().contiguous()
    
    # Initialize GNN model with correct dimensions
    global _gnn_model
    if _gnn_model is None:
        _gnn_model = StrategicGNN(
            node_features=20,  # Updated from 25 after removing deception pattern features
            message_dim=node_message_embeddings.shape[1],
            embedding_dim=256
        ).to(device)
        _gnn_model.eval()
    
    # Convert to tensors
    node_tensor = torch.FloatTensor(node_features).to(device)
    msg_tensor = torch.FloatTensor(node_message_embeddings).to(device)
    edge_index = edge_index.to(device)
    
    # Generate embeddings
    with torch.no_grad():
        strategic_embeddings = _gnn_model(node_tensor, msg_tensor, edge_index)
    
    # Return embeddings for both sender and receiver
    return strategic_embeddings.cpu().numpy()  # Shape: (2, 256) - [sender_emb, receiver_emb]

