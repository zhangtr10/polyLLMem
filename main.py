import numpy as np

import json
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold  # Changed to StratifiedKFold
from dataset import *
from train import * 
from model import *


logger = setup_logger()
logger.info("Starting training with K-Fold Cross Validation")

# Create checkpoint directory
os.makedirs('checkpoints', exist_ok=True)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
property='Tg'
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and split data


with open(f'./train_{property}_data.json', 'r') as f:
    all_data = json.load(f)
polymers = all_data['polymers']
with open(f'./test_{property}_data.json', 'r') as f:

    all_data = json.load(f)
polymers_test = all_data['polymers']

logger.info('#' * 40 + "  Settings  " + '#' * 40)
# Configuration
config = {
    'hidden_size': 1536,  'rank': 16,  'alpha': 32, 
    'dropout': 0.1,    'batch_size': 16,
    'num_epochs': 1000,   'lr': 1e-4,    'delta': 30, 
    'weight_decay': 0.00001,  'patience': 30,  'min_delta': 0,
}

lines = [
    ["hidden_size", "rank", "alpha", "dropout", "batch_size"],
    ["num_epochs", "delta", "lr", "weight_decay", "patience", "min_delta"]
]

for keys in lines:
    header = " | ".join(f"{k:<12}" for k in keys)
    values = " | ".join(f"{config[k]:<12}" for k in keys)
    logger.info(header)
    logger.info("-" * len(header))
    logger.info(values)


logger.info('\n\n\n')
logger.info('#' * 40 + "  Progress  " + '#' * 40)

#tg_values = np.array([float(polymer['metadata']['glass_transition_temp']) for polymer in polymers])
# Define number of bins (adjust as necessary)
normalizer = None
# Clean tg_values and store valid polymer entries
tg_values_cleaned = []
polymers_cleaned = []

for polymer in polymers:
    tg = polymer['metadata']['glass_transition_temp']
    try:
        tg_float = float(tg)
        tg_values_cleaned.append(tg_float)
        polymers_cleaned.append(polymer)  # keep the aligned polymer
    except (TypeError, ValueError):
        continue

tg_values =   np.array(tg_values_cleaned)
polymers = np.array(polymers_cleaned)
# Use StratifiedKFold with the binned Tg values.
# num_bins = 5
# bins = np.linspace(tg_values.min(), tg_values.max(), num_bins + 1)
# # Assign each Tg value to a bin.
# tg_bins = np.digitize(tg_values, bins)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)



kf = KFold(n_splits=5, shuffle=True, random_state=99)

mae_scores = []
r2_scores = []
mae_scores_val = []
r2_scores_val = []
test_dataset = PolymerDataset(polymers_test, gpt_embedding_file=f'Llama_polymer_embeddings_{property}_just_psmile.pickle',
                                conf_embedding_file=f'uni_polymer_embeddings_{property}_just_smile.pickle',
                                normalizer=normalizer, do_normalize=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0, drop_last=True
)
#for fold, (train_idx, val_idx) in enumerate(skf.split(polymers, tg_bins)):

for fold, (train_idx, val_idx) in enumerate(kf.split(polymers)):
    logger.info(f"Starting Fold {fold + 1}")
    train_data = [polymers[i] for i in train_idx]
    val_data = [polymers[i] for i in val_idx]
    

    train_dataset = PolymerDataset(train_data, gpt_embedding_file=f'Llama_polymer_embeddings_{property}_just_psmile.pickle',
                                    conf_embedding_file=f'uni_polymer_embeddings_{property}_just_smile.pickle',
                                    normalizer=normalizer, do_normalize=True)
    val_dataset = PolymerDataset(val_data, gpt_embedding_file=f'Llama_polymer_embeddings_{property}_just_psmile.pickle',
                                    conf_embedding_file=f'uni_polymer_embeddings_{property}_just_smile.pickle',
                                    normalizer=train_dataset.normalizer, do_normalize=False)
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0, drop_last=True
    )
    
    model = PolymerModel_cat(
        gpt_model=None,
        unimol_model=None,
        hidden_size=config['hidden_size'],
        rank=config['rank'],
        alpha=config['alpha'],
        dropout=config['dropout']
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    criterion_dict = {
        'mse': nn.MSELoss().to(device),
        'mae': nn.L1Loss().to(device),
        'hub': nn.HuberLoss(delta=config['delta']).to(device)
    }
    
    early_stopping = EarlyStopping(patience=config['patience'], min_delta=config['min_delta'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=3
    )
    
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_dict, device, epoch, config
        )
        
        val_metrics = validate(model, val_loader, criterion_dict, device, config, train_dataset.normalizer)
        
        scheduler.step(val_metrics['loss'])
        
        if val_metrics['loss'] < best_val_loss:
            log_metrics(logger, epoch, train_metrics, val_metrics, config, True)


            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config
            }, f'checkpoints/best_model_fold{fold}_{property}_v4.pth')
        else:
            log_metrics(logger, epoch, train_metrics, val_metrics, config, False)

        early_stop, _, _ = early_stopping(
            val_metrics['loss'],
            model=model,
            dump_dir='checkpoints',
            fold=0,
            epoch=epoch
        )
        
        if early_stop:
            logger.info("Early stopping triggered")
            break

    
    logger.info(f"Training completed for Fold {fold + 1}")
    
    state_dict = torch.load(f'checkpoints/best_model_fold{fold}_{property}_v4.pth')
    model.load_state_dict(state_dict['model_state_dict'])

    test_metrics = test(model, test_dataset, criterion_dict, device, train_dataset.normalizer)
    
    mae_scores.append(test_metrics['mae'])
    r2_scores.append(test_metrics['r2'])

    val_metrics = test(model, val_dataset, criterion_dict, device, train_dataset.normalizer)
    
    mae_scores_val.append(val_metrics['mae'])
    r2_scores_val.append(val_metrics['r2'])
# Compute final statistics
mean_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)
mean_r2, std_r2 = np.mean(r2_scores), np.std(r2_scores)

mean_mae_val, std_mae_val = np.mean(mae_scores_val), np.std(mae_scores_val)
mean_r2_val, std_r2_val = np.mean(r2_scores_val), np.std(r2_scores_val)
logger.info(f"Final Results after 5-Fold Cross Test:")
logger.info(f"MAE: {mean_mae:.4f} ± {std_mae:.4f}")
logger.info(f"R²: {mean_r2:.4f} ± {std_r2:.4f}")
logger.info(f"Final Results after 5-Fold Cross Validation:")
logger.info(f"MAE: {mean_mae_val:.4f} ± {std_mae_val:.4f}")
logger.info(f"R²: {mean_r2_val:.4f} ± {std_r2_val:.4f}")