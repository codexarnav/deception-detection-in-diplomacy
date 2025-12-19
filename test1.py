import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import numpy as np
from torch.nn import MultiheadAttention
import faiss
from sentence_transformers import SentenceTransformer

from gnn import get_strategic_embeddings


# ----------------------------------------------------------------------
# LOAD SENTENCE TRANSFORMER
# ----------------------------------------------------------------------

model_path = 'test_allminilm_finetuned-20250829T234732Z-1-001/test_allminilm_finetuned'
print(f"Loading fine-tuned embedding model from: {model_path}")
text_model = SentenceTransformer(model_path)


def generalEmbeddings(Message):
    return text_model.encode(Message, normalize_embeddings=True)


# ----------------------------------------------------------------------
# BALANCED BATCH SAMPLER
# ----------------------------------------------------------------------

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size=32, minority_ratio=0.25):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.minority_ratio = minority_ratio

        self.minority_indices = np.where(self.labels == 1)[0].tolist()
        self.majority_indices = np.where(self.labels == 0)[0].tolist()

        self.minority_per_batch = int(batch_size * minority_ratio)
        self.majority_per_batch = batch_size - self.minority_per_batch

    def __iter__(self):
        minority_pool = self.minority_indices.copy()
        majority_pool = self.majority_indices.copy()

        random.shuffle(minority_pool)
        random.shuffle(majority_pool)

        m_ptr = 0
        M_ptr = 0

        while True:
            if m_ptr + self.minority_per_batch > len(minority_pool):
                random.shuffle(minority_pool)
                m_ptr = 0
            if M_ptr + self.majority_per_batch > len(majority_pool):
                random.shuffle(majority_pool)
                M_ptr = 0

            batch = (
                minority_pool[m_ptr:m_ptr+self.minority_per_batch] +
                majority_pool[M_ptr:M_ptr+self.majority_per_batch]
            )
            random.shuffle(batch)

            m_ptr += self.minority_per_batch
            M_ptr += self.majority_per_batch

            yield batch

    def __len__(self):
        return len(self.labels) // self.batch_size



# ----------------------------------------------------------------------
# DATASET CLASS
# ----------------------------------------------------------------------

class DiplomacyDataset(Dataset):
    def __init__(self, strategic_embs, text_embs, labels):
        self.strategic = strategic_embs
        self.text = text_embs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.strategic[idx]),
            torch.FloatTensor(self.text[idx]),
            torch.FloatTensor([self.labels[idx]])
        )



# ----------------------------------------------------------------------
# MODEL ARCHITECTURE
# ----------------------------------------------------------------------

class EmbeddingFusion(nn.Module):
    def __init__(self, strategic_dim=256, text_dim=256, fusion_dim=512):
        super(EmbeddingFusion, self).__init__()

        self.strategic_proj = nn.Linear(strategic_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        self.cross_attention = MultiheadAttention(
            fusion_dim, num_heads=8, batch_first=True
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(fusion_dim)
        )

    def forward(self, strategic_emb, text_emb):
        strategic_proj = self.strategic_proj(strategic_emb).unsqueeze(1)
        text_proj = self.text_proj(text_emb).unsqueeze(1)

        attn_out, _ = self.cross_attention(
            strategic_proj, text_proj, text_proj
        )

        fused = torch.cat(
            [strategic_proj.squeeze(1), attn_out.squeeze(1)], dim=-1
        )
        return self.fusion_layer(fused)



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return loss.mean()



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

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.consistency_head = nn.Linear(fusion_dim, 1)
        self.confidence_head = nn.Linear(fusion_dim, 1)

    def forward(self, strategic_emb, text_emb, training=True):
        fused = self.fusion(strategic_emb, text_emb)

        epistemic, aleatoric = self.uncertainty(fused)

        deception_logits = self.classifier(fused)
        consistency_score = torch.sigmoid(self.consistency_head(fused))
        confidence_score = torch.sigmoid(self.confidence_head(fused))

        if training:
            return {
                'logits': deception_logits,
                'consistency': consistency_score,
                'confidence': confidence_score,
                'epistemic': epistemic,
                'aleatoric': aleatoric
            }

        # Monte-Carlo dropout
        self.train()
        preds = []
        for _ in range(self.n_monte_carlo):
            preds.append(self.classifier(fused))
        self.eval()

        preds = torch.stack(preds)
        mean_pred = preds.mean(dim=0)
        epistemic_unc = preds.var(dim=0)

        return {
            'logits': mean_pred,
            'consistency': consistency_score,
            'confidence': confidence_score,
            'epistemic_uncertainty': epistemic_unc,
            'aleatoric_uncertainty': aleatoric
        }



# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------

class DeceptionPipeline:
    def __init__(self, strategic_dim=256, text_dim=256):
        self.model = DeceptionDetector(strategic_dim, text_dim)
        self.loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    def train_step(self, strategic_batch, text_batch, target_batch, optimizer):
        self.model.train()

        optimizer.zero_grad()

        outputs = self.model(strategic_batch, text_batch, training=True)
        logits = outputs["logits"]

        loss = self.loss_fn(logits, target_batch)

        loss.backward()
        optimizer.step()

        return loss.item()

    def predict(self, strategic_emb, text_emb):
        self.model.eval()

        with torch.no_grad():
            strategic_t = torch.FloatTensor(strategic_emb)
            text_t = torch.FloatTensor(text_emb)

            out = self.model(strategic_t, text_t, training=False)

            deception_score = torch.sigmoid(out['logits']).squeeze().numpy()
            epistemic = out['epistemic_uncertainty'].squeeze().numpy()
            aleatoric = out['aleatoric_uncertainty'].squeeze().numpy()

            hallucination = (
                epistemic * 0.4 +
                aleatoric * 0.3 +
                (1 - out['consistency']).squeeze().numpy() * 0.3
            )

            return {
                "deception_score": deception_score,
                "epistemic_uncertainty": epistemic,
                "aleatoric_uncertainty": aleatoric,
                "hallucination_score": hallucination
            }




# ----------------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------------

def train_pipeline(strategic_embs, text_embs, labels, epochs=5, batch_size=32, lr=1e-4):
    dataset = DiplomacyDataset(strategic_embs, text_embs, labels)

    sampler = BalancedBatchSampler(labels, batch_size=batch_size, minority_ratio=0.25)

    loader = DataLoader(dataset, batch_sampler=sampler)

    pipeline = DeceptionPipeline()
    optimizer = torch.optim.Adam(pipeline.model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for strategic_batch, text_batch, target_batch in loader:
            loss = pipeline.train_step(
                strategic_batch,
                text_batch,
                target_batch,
                optimizer
            )
            total_loss += loss

        print(f"Epoch {epoch+1}/{epochs} — Loss = {total_loss:.4f}")

    return pipeline



# ----------------------------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------------------------

def main():
    senders = ["France"]
    receivers = ["Germany"]
    messages = [
        """I propose we form an alliance against Austria
        I agree but we should be cautious
        Russia is planning to attack us"""
    ]

    game_contexts = [
        {'game_score': 120, 'game_score_delta': 10, 'absolute_message_index': 5, 'relative_message_index': 2}
    ]

    # Get strategic embeddings
    pipeline = DeceptionPipeline()
    strategic_emb = np.array([
        get_strategic_embeddings(senders[0], receivers[0], messages[0], game_contexts[0])[0]
    ])

    # Text embeddings
    text_emb = np.array([generalEmbeddings(messages[0])])

    result = pipeline.predict(strategic_emb, text_emb)

    print("Deception Score:", result['deception_score'])
    print("Uncertainty:", result['epistemic_uncertainty'])

    return pipeline


if __name__ == "__main__":
    pipeline = main()
