# Diplomacy Deception Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning pipeline for detecting deception in diplomatic communications using multimodal fusion of text embeddings and graph neural networks.

## 🎯 Overview

This system analyzes diplomatic game logs from the board game Diplomacy to predict deception patterns in strategic communications. By combining natural language processing with graph-based relational modeling, we achieve uncertainty-aware multi-class deception detection that accounts for the complex social dynamics inherent in strategic negotiations.

### Key Features

- **Multi-class deception detection** with nuanced outcome categories
- **Multimodal fusion** combining text embeddings and strategic graph embeddings  
- **Uncertainty quantification** using Monte-Carlo dropout and auxiliary supervision
- **Per-message graph modeling** for tractable relational analysis
- **Robust label handling** for noisy diplomatic communication data

## 🏗️ Architecture

The system implements a three-pathway architecture:

```
Raw Messages → Text Embedding (Transformer) ┐
                                            ├→ Fusion Classifier → Predictions + Uncertainty
Strategic Context → Graph Embedding (GNN) ┘
```

### Components

1. **Language Pathway**: SentenceTransformer embeddings with handcrafted linguistic features
2. **Strategic GNN**: Graph Attention Network modeling sender-receiver interactions
3. **Fusion Classifier**: Multi-head architecture with uncertainty quantification

## 📊 Dataset

The system processes diplomatic message exchanges with the following features:

- **Messages**: Raw diplomatic text communications
- **Players**: Sender/receiver identifiers and country assignments
- **Labels**: Boolean annotations for sender and receiver deception
- **Temporal**: Message indices, seasons, years, and game scores
- **Strategic**: Game state and score deltas

### Label Mapping

The system maps boolean label pairs to interpretable deception states:

| Sender Label | Receiver Label | Deception State |
|--------------|----------------|-----------------|
| True | True | `no_deception` |
| True | False | `no_deception` |
| False | True | `successful_deception` |
| False | False | `successful_deception` |

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision
pip install torch-geometric
pip install sentence-transformers
pip install scikit-learn pandas numpy
pip install matplotlib seaborn
```

### Basic Usage

```python
from train_pipeline import train_model
from gnn import StrategicGNN
from test import MultiClassDeceptionDetector

# Train the model
model, history = train_model(
    dataset_path="final_dataset1.csv",
    config_path="config.json"
)

# Load trained model for inference
detector = MultiClassDeceptionDetector.load_from_checkpoint("artifacts/best_model.pth")
predictions, uncertainty = detector.predict_with_uncertainty(messages)
```

### Training Pipeline

```bash
python train_pipeline.py --dataset final_dataset1.csv --epochs 20 --batch_size 32
```

## 📁 Repository Structure

```
├── gnn.py                    # Strategic GNN and text processing components
├── train_pipeline.py         # End-to-end training pipeline
├── test.py                   # Fusion classifier and uncertainty modules
├── utils.py                  # Data parsing and label mapping utilities
├── config.json               # Experiment configuration
├── artifacts/                # Trained model artifacts
│   ├── best_model.pth
│   ├── label_encoder.pkl
│   └── training_history.pkl
└── README.md
```

## ⚙️ Configuration

Key hyperparameters in `config.json`:

```json
{
  "batch_size": 32,
  "learning_rate": 1e-4,
  "n_epochs": 20,
  "fusion_dim": 512,
  "strategic_dim": 256,
  "text_dim": 256,
  "loss_weights": {
    "alpha": 1.0,
    "beta": 0.5, 
    "gamma": 0.3
  }
}
```

## 🎯 Model Performance

The system provides:

- **Multi-class predictions** for deception outcome categories
- **Epistemic uncertainty** via Monte-Carlo dropout
- **Aleatoric uncertainty** through auxiliary heads
- **Calibrated confidence** scores for human-in-the-loop workflows

## 🔬 Technical Details

### Feature Engineering

**Node-level features** (per player):
- Message embedding aggregations
- Communication volume and ratios
- Score trends and centrality measures
- Activity patterns and temporal dynamics

**Edge-level features** (per interaction):
- Semantic message embeddings
- Interaction sentiment analysis
- Handcrafted textual features (word counts, punctuation, deception lexicon matches)

**Temporal features**:
- Message positioning and game state indicators
- Score deltas and strategic timing signals

### Graph Neural Network

- **Architecture**: 3-layer Graph Attention Network (GATv2)
- **Input**: 25-dimensional node features + message embeddings
- **Design**: Per-message 2-node graphs for computational tractability
- **Output**: 256-dimensional strategic embeddings

## ⚠️ Limitations

1. **Local graphs only**: Uses 2-node per-message graphs rather than full temporal dynamics
2. **Label noise**: Some ambiguous annotations remain without human adjudication  
3. **Heuristic mapping**: Boolean-to-multiclass conversion uses pragmatic heuristics
4. **Limited explainability**: Confidence signals available but no token-level explanations

## 🔮 Future Work

### Short-term
- Implement session-level temporal GNNs (TGN/Temporal Attention)
- Human-in-the-loop active learning for label refinement
- Add explainability modules with attention visualization

### Long-term  
- Deploy inference pipeline with human review workflows
- Explore reinforcement learning for adversarial strategy modeling
- Extend to other strategic communication domains

## 📚 References

- Peskov, D. (2020). "It Takes Two to Lie: One to Lie, and One to Listen." ACL.
- Niculae, V. et al. (2015). "Linguistic harbingers of betrayal." ACL.
- Wongkamjan, W. et al. (2024-2025). "Detecting deception in negotiations using counterfactual RL." arXiv.

## 🤝 Contributing

We welcome contributions! The system is designed to be modular and extensible. Key areas for improvement:

- Enhanced temporal modeling
- Alternative fusion architectures  
- Improved uncertainty calibration
- Cross-domain adaptation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built during a machine learning hackathon focusing on strategic communication analysis. Special thanks to the Diplomacy gaming community for providing rich datasets of strategic interactions.

---

**Note**: This system is designed for research purposes in computational social science and game theory. It demonstrates techniques for analyzing strategic communication patterns in controlled gaming environments.
