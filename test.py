import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import MultiheadAttention
import faiss

from sentence_transformers import SentenceTransformer
import numpy as np

# Import GNN functions
from gnn import get_strategic_embeddings

# 1. Define the path to your saved model directory
model_path = 'test_allminilm_finetuned-20250829T234732Z-1-001/test_allminilm_finetuned' # Or wherever you saved it

# 2. Load the fine-tuned model
print(f"Loading fine tunned embedding model from: {model_path}")
model = SentenceTransformer(model_path)

# 
def generalEmbeddings(Message):
    embedding =model.encode(Message, normalize_embeddings=True)
    return embedding

class EmbeddingFusion(nn.Module):
    def __init__(self, strategic_dim=256, text_dim=256, fusion_dim=512):
        super(EmbeddingFusion, self).__init__()
        
        # Projection layers to match dimensions
        self.strategic_proj = nn.Linear(strategic_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        
        # Cross-attention for fusion
        self.cross_attention = MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, strategic_emb, text_emb):
        # Project embeddings to same dimension
        strategic_proj = self.strategic_proj(strategic_emb).unsqueeze(1)  # [B, 1, D]
        text_proj = self.text_proj(text_emb).unsqueeze(1)  # [B, 1, D]
        
        # Cross-attention between strategic and textual
        attn_out, _ = self.cross_attention(strategic_proj, text_proj, text_proj)
        
        # Concatenate and fuse
        fused = torch.cat([strategic_proj.squeeze(1), attn_out.squeeze(1)], dim=-1)
        return self.fusion_layer(fused)
class FocalLoss(nn.Module):
    def __init__(self,alpha,gamma,reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.reduction=reduction

    def forward(self,inputs,targets):
        BCE_loss=nn.functional.binary_cross_entropy_with_logits(inputs,targets,reduction='none')
        pt=torch.exp(-BCE_loss)
        focal_loss=self.alpha*(1-pt)**self.gamma*BCE_loss
        if self.reduction=='mean':
            return focal_loss.mean()
        elif self.reduction=='sum':
            return focal_loss.sum()
        else:
            return focal_loss
class UncertaintyQuantification(nn.Module):
    def __init__(self, input_dim=512):
        super(UncertaintyQuantification, self).__init__()
        self.epistemic_head = nn.Linear(input_dim, input_dim // 2)
        self.aleatoric_head = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        epistemic = F.relu(self.epistemic_head(x))
        aleatoric = F.softplus(self.aleatoric_head(x))
        return epistemic, aleatoric

class DeceptionDetector(nn.Module):
    def __init__(self, strategic_dim=256, text_dim=256, fusion_dim=512, n_monte_carlo=10):
        super(DeceptionDetector, self).__init__()
        
        self.fusion = EmbeddingFusion(strategic_dim, text_dim, fusion_dim)
        self.uncertainty = UncertaintyQuantification(fusion_dim)
        self.n_monte_carlo = n_monte_carlo
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
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
            deception_score = self.classifier(fused_emb)
            consistency_score = torch.sigmoid(self.consistency_head(fused_emb))
            confidence_score = torch.sigmoid(self.confidence_head(fused_emb))
            
            return {
                'deception_score': deception_score,
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
                pred = self.classifier(fused_emb)
                predictions.append(pred)
            
            self.eval()
            predictions = torch.stack(predictions)
            
            # Calculate uncertainty metrics
            mean_pred = predictions.mean(dim=0)
            epistemic_uncertainty = predictions.var(dim=0)
            
            consistency_score = torch.sigmoid(self.consistency_head(fused_emb))
            confidence_score = torch.sigmoid(self.confidence_head(fused_emb))
            
            return {
                'deception_score': mean_pred,
                'epistemic_uncertainty': epistemic_uncertainty,
                'aleatoric_uncertainty': aleatoric.squeeze(),
                'consistency_score': consistency_score,
                'confidence_score': confidence_score,
                'prediction_variance': epistemic_uncertainty
            }



class DeceptionPipeline:
    def __init__(self, model_path=None, strategic_dim=256, text_dim=256):
        self.model = DeceptionDetector(strategic_dim, text_dim)
        self.loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
    def get_strategic_embeddings_batch(self, senders, receivers, messages, game_contexts=None):
        """Get strategic embeddings for a batch of messages using GNN"""
        batch_embeddings = []
        
        for i in range(len(senders)):
            sender = senders[i]
            receiver = receivers[i]
            message = messages[i]
            context = game_contexts[i] if game_contexts else None
            
            # Get strategic embeddings from GNN (returns [sender_emb, receiver_emb])
            embeddings = get_strategic_embeddings(sender, receiver, message, context)
            
            # Use sender embedding for this pipeline
            batch_embeddings.append(embeddings[0])  # sender embedding
        
        return np.array(batch_embeddings)
    
    def load_strategic_embeddings(self, index_path):
        """Load FAISS index with strategic embeddings"""
        self.strategic_index = faiss.read_index(index_path)
        return self.strategic_index
    
    def predict(self, strategic_emb, text_emb):
        """
        Main inference function
        Args:
            strategic_emb: Strategic embedding from GNN [batch_size, strategic_dim]
            text_emb: Text embedding from fine-tuned model [batch_size, text_dim]
        Returns:
            dict with deception scores, uncertainties, and hallucination indicators
        """
        self.model.eval()
        with torch.no_grad():
            strategic_tensor = torch.FloatTensor(strategic_emb)
            text_tensor = torch.FloatTensor(text_emb)
            
            results = self.model(strategic_tensor, text_tensor, training=False)
            
            # Calculate hallucination score (high uncertainty + low consistency)
            hallucination_score = (
                results['epistemic_uncertainty'].squeeze() * 0.4 +
                results['aleatoric_uncertainty'] * 0.3 +
                (1 - results['consistency_score'].squeeze()) * 0.3
            )
            
            return {
                'deception_score': results['deception_score'].squeeze().numpy(),
                'hallucination_score': hallucination_score.numpy(),
                'epistemic_uncertainty': results['epistemic_uncertainty'].squeeze().numpy(),
                'aleatoric_uncertainty': results['aleatoric_uncertainty'].numpy(),
                'consistency_score': results['consistency_score'].squeeze().numpy(),
                'confidence_score': results['confidence_score'].squeeze().numpy()
            }
    
    def train_step(self, strategic_emb, text_emb, targets, optimizer):
        """Single training step"""
        self.model.train()
        
        strategic_tensor = torch.FloatTensor(strategic_emb)
        text_tensor = torch.FloatTensor(text_emb)
        target_tensor = torch.LongTensor(targets)
        
        optimizer.zero_grad()
        
        predictions = self.model(strategic_tensor, text_tensor, training=True)
        losses = self.loss_fn(predictions, target_tensor)
        
        losses['total_loss'].backward()
        optimizer.step()
        
        return losses

# Usage Example
def main():
    # Initialize pipeline
    pipeline = DeceptionPipeline(strategic_dim=256, text_dim=256)
    
    # Example usage with GNN embeddings
    senders = ["France"]
    receivers = ["Germany"]
    messages = [
        """I propose we form an alliance against Austria
        I agree but we should be cautious
        Russia is planning to attack us"""
    ]
    
    game_contexts = [
        {'game_score': 120, 'game_score_delta': 10, 'absolute_message_index': 5, 'relative_message_index': 2},
        {'game_score': 100, 'game_score_delta': -5, 'absolute_message_index': 6, 'relative_message_index': 3},
        {'game_score': 90, 'game_score_delta': -10, 'absolute_message_index': 7, 'relative_message_index': 1}
    ]
    
    # Get strategic embeddings from GNN
    strategic_emb = pipeline.get_strategic_embeddings_batch(senders, receivers, messages, game_contexts)
    
    # Get text embeddings
    text_emb = np.array([generalEmbeddings(msg) for msg in messages])
    
    # Make predictions
    results = pipeline.predict(strategic_emb, text_emb)
    
    print(f"Deception Scores: {results['deception_score']}")
    print(f"Hallucination Scores: {results['hallucination_score']}")
    print(f"Uncertainties: {results['epistemic_uncertainty']}")
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()