import optuna
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import functionalities from train1.py
# ensuring train1.py is in the python path or same directory
from train1 import (
    load_and_preprocess_data,
    generate_embeddings,
    prepare_labels,
    DeceptionDataset,
    MultiClassDeceptionDetector,
    FocalLoss,
    train_model
)

def objective(trial):
    # 1. Hyperparameters to tune
    k_neighbors = trial.suggest_int('k_neighbors', 3, 10)
    
    # Tuning learning rate and Focal Loss parameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    alpha = trial.suggest_float('alpha', 0.1, 0.9)
    gamma = trial.suggest_float('gamma', 0.5, 5.0)
    
    # We can tune sampling strategy (ratio of minority/majority)
    # Since we have multi-class, 'auto' simply resamples all except majority to equal majority
    # If we want to tune specific ratios, it's more complex with multi-class SMOTE.
    # For now, let's stick to 'auto' or 'not majority' but mainly tune k_neighbors 
    # and maybe a sampling text/strategy mix ratio if we had one. 
    # The user specifically asked for SMOTE tuning.
    
    # Let's try to tune a float ratio if possible, but SMOTE implementation 
    # for 'sampling_strategy' as a float only works for binary classification usually.
    # For multiclass, it accepts a dict. 
    # To keep it simple and robust, we will look at k_neighbors primarily,
    # and maybe we can try tweaking the 'kind' if we were using SVM SMOTE, but here it's regular SMOTE.
    
    # Let's just focus on k_neighbors for this architecture as requested.
    
    # Global data (loaded once outside to save time)
    global text_emb, strat_emb, labels, label_encoder, device, config
    
    # 2. Fuse Embeddings
    fused = np.concatenate([text_emb, strat_emb], axis=1)
    
    # 3. Apply SMOTE with suggested params
    try:
        sm = SMOTE(k_neighbors=k_neighbors, random_state=42)
        fused_resampled, labels_resampled = sm.fit_resample(fused, labels)
    except ValueError as e:
        # k_neighbors might be too large for some very small minority classes
        print(f"Pruning trial due to SMOTE error: {e}")
        raise optuna.exceptions.TrialPruned()

    # 4. Un-fuse
    text_dim = text_emb.shape[1]
    X_text = fused_resampled[:, :text_dim]
    X_strat = fused_resampled[:, text_dim:]
    y = labels_resampled

    # 5. Split (Train/Val only for tuning)
    X_text_train, X_text_val, X_strat_train, X_strat_val, y_train, y_val = train_test_split(
        X_text, X_strat, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 6. Dataloaders
    train_dataset = DeceptionDataset(X_text_train, X_strat_train, y_train)
    val_dataset = DeceptionDataset(X_text_val, X_strat_val, y_val)
    
    # Weighted sampler
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = class_weights[y_train]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        # Using a smaller batch size for tuning can be faster? keeping consistent for now
        sampler=WeightedRandomSampler(sample_weights, len(sample_weights))
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # 7. Model Setup (Re-init every trial)
    model = MultiClassDeceptionDetector(
        strategic_dim=config['strategic_dim'],
        text_dim=config['text_dim'],
        fusion_dim=config['fusion_dim'],
        n_classes=len(label_encoder.classes_)
    ).to(device)
    
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 8. Train for fewer epochs for tuning speed
    n_tuning_epochs = 5 
    
    # We need a modified train loop that returns val_f1 directly or use the existing one
    # The existing train_model returns a history dict
    history = train_model(
        train_loader, val_loader, model, criterion, optimizer, scheduler,
        device, n_epochs=n_tuning_epochs, save_dir="./optuna_checkpoints"
    )
    
    return history['best_val_f1']

if __name__ == "__main__":
    # Config matching train1.py
    config = {
        'csv_path': 'data/final_dataset1.csv',
        'model_path': 'test_allminilm_finetuned-20250829T234732Z-1-001/test_allminilm_finetuned',
        'strategic_dim': 256,
        'text_dim': 256,
        'fusion_dim': 512,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print("Loading Data for Optuna Study...")
    device = config['device']
    
    # Load once
    df = load_and_preprocess_data(config['csv_path'])
    text_emb, strat_emb = generate_embeddings(df, config['model_path'])
    labels, label_encoder = prepare_labels(df)
    
    print("Starting Optuna Study...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
