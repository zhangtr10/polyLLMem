
import logging
from sklearn.metrics import r2_score
import torch
import numpy as np
import os
import datetime
import shutil
from pathlib import Path

## EARLY STOPPING TRAINING
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model=None, dump_dir=None, fold=None, epoch=None):

        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None and dump_dir is not None:
                self._save_model(model, dump_dir, fold)
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and dump_dir is not None:
                self._save_model(model, dump_dir, fold)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                #if epoch is not None:
                #    print(f'Early stopping at epoch: {epoch+1}')
                self.early_stop = True
                
        return self.early_stop, self.best_loss, self.counter
    
    def _save_model(self, model, dump_dir, fold):
        """Save model state"""
        os.makedirs(dump_dir, exist_ok=True)
        info = {'model_state_dict': model.state_dict()}
        torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))

## SAVE AND LOAD
def save_checkpoint(state, is_best, filename='checkpoint.pth', best_filename='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
    else:
        raise FileNotFoundError(f"No checkpoint found at {filename}")

## LOGGER
def setup_logger(log_dir='logs'):
    """Set up logger to write to file and console"""
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create timestamp for unique log file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'training_{timestamp}.log'
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger(__name__)

def log_metrics(logger, epoch, train_metrics, val_metrics, config, is_best_model):
    """Log training and validation metrics in a compact table format with a header"""
    if epoch == 0:  # Print the header only for the first epoch
        logger.info(f"{'Epoch':<6} | {'Train Loss':<10} | {'Train MAE':<10} | {'Val Loss':<10} | {'Val MAE':<10} | {'Val R2':<10} | {'Best':<3}")
        logger.info("-" * 90)
    
    best_model_status = "√" if is_best_model else "x"
    logger.info(
        f"{epoch+1:<6} | {train_metrics['loss']:<10.4f} | {train_metrics['mae_loss']:<10.4f} | {val_metrics['loss']:<10.4f} | {val_metrics['mae']:<10.4f} | {val_metrics['r2']:<10.4f} | {best_model_status:<3}"
    )


def train_epoch(model, train_loader, optimizer, criterion_dict, device, epoch,config, save_freq=100):
    # Debug prints
    #print(f"Model device: {next(model.parameters()).device}")
    #print(f"Device being used: {device}")
    for name, criterion in criterion_dict.items():
        if hasattr(criterion, 'to'):
            criterion_dict[name] = criterion.to(device)

    model.train()
    total_loss = 0
    total_mse = 0
    total_nce = 0
    total_mae  = 0
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        gpt_embed = batch['gpt_embed'].to(device)
        conf_embed = batch['conf_embed'].to(device) 
        tg_values = batch['tg_value'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(gpt_embed, conf_embed)
        # Calculate losses
        mse_loss = criterion_dict['mse'](outputs['prediction'].squeeze(), tg_values)
        mae_loss = criterion_dict['mae'](outputs['prediction'].squeeze(), tg_values)
        hub_loss = criterion_dict['hub'](outputs['prediction'].squeeze(), tg_values)
        # Combined loss
        #loss = mse_loss +  config['scale']*nce_loss
        loss = hub_loss
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_mse += mse_loss.item()
        #total_nce += nce_loss.item()
        total_mae += mae_loss.item()
        # Save checkpoint
        if (batch_idx + 1) % save_freq == 0:
            state = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / (batch_idx + 1),
                'batch_metrics': {
                    'mse': total_mse / (batch_idx + 1)
                    #'nce': total_nce / (batch_idx + 1)
                }
            }
            save_checkpoint(
                state, 
                is_best=False,
                filename=f'checkpoints/epoch_{epoch}_batch_{batch_idx}.pth'
            )
        

    return {
        'loss': total_loss/len(train_loader),
        'mse_loss': total_mse/len(train_loader),
        #'nce_loss': total_nce/len(train_loader),
        'mae_loss': total_mae/len(train_loader)
    }

## VALIDATE
def validate(model, val_loader, criterion_dict, device, config, normalizer=None):
    model.eval()
    total_loss, total_mse, total_mae, sample = 0, 0, 0, 0
    predictions, targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to the device
            gpt_embed = batch['gpt_embed'].to(device)
            conf_embed = batch['conf_embed'].to(device)
            tg_values = batch['tg_value'].to(device).view(-1)

            # Get model outputs and compute predictions
            predictions_batch = model(gpt_embed, conf_embed)['prediction'].view(-1)

            # Denormalize predictions and targets if normalizer is provided
            if normalizer is not None:
                denorm_predictions = normalizer.denormalize(predictions_batch.cpu().numpy())
                denorm_targets = normalizer.denormalize(tg_values.cpu().numpy())
            else:
                denorm_predictions = predictions_batch.cpu().numpy()
                denorm_targets = tg_values.cpu().numpy()

            # Convert denormalized values to tensors for loss computation
            denorm_predictions_tensor = torch.tensor(denorm_predictions).to(device)
            denorm_targets_tensor = torch.tensor(denorm_targets).to(device)

            # Compute losses using denormalized values
            mse_loss = criterion_dict['mse'](denorm_predictions_tensor, denorm_targets_tensor)
            mae_loss = criterion_dict['mae'](denorm_predictions_tensor, denorm_targets_tensor)
            hub_loss = criterion_dict['hub'](denorm_predictions_tensor, denorm_targets_tensor)

            # Accumulate metrics
            batch_size = tg_values.size(0)
            total_loss += hub_loss.item()
            total_mse += mse_loss.item()
            total_mae += mae_loss.item() * batch_size

            # Store predictions and targets
            predictions.extend(denorm_predictions)
            targets.extend(denorm_targets)
            sample += batch_size

    # Convert results to numpy arrays
    predictions, targets = np.array(predictions), np.array(targets)

    return {
        'loss': total_loss / len(val_loader),
        'mse_loss': total_mse / len(val_loader),
        'mae': total_mae / sample,
        'r2': r2_score(targets, predictions),
        'predictions': predictions,
        'targets': targets,
    }
## Test and Plot
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600

def test(model, dataset, criterion_dict, device, normalizer=None):
    model.eval()
    total_loss, total_mse, total_mae, sample = 0, 0, 0, len(dataset)
    predictions, targets = [], []

    with torch.no_grad():
        for sample_data in dataset:
            # Move data to the device
            gpt_embed = sample_data['gpt_embed'].to(device).unsqueeze(0)
            conf_embed = sample_data['conf_embed'].to(device).unsqueeze(0)
            tg_value = sample_data['tg_value'].to(device)

            # Model prediction
            outputs = model(gpt_embed, conf_embed)
            prediction = outputs['prediction'].squeeze()

            # Denormalize if a normalizer is provided
            if normalizer is not None:
                denorm_pred = normalizer.denormalize(prediction.cpu().numpy())
                denorm_tg = normalizer.denormalize(tg_value.cpu().numpy())
            else:
                denorm_pred = prediction.cpu().numpy()
                denorm_tg = tg_value.cpu().numpy()

            # Convert denormalized values to tensors for loss computation
            denorm_pred_tensor = torch.tensor(denorm_pred).to(device)
            denorm_tg_tensor = torch.tensor(denorm_tg).to(device)

            # Compute losses using denormalized values
            mse_loss = criterion_dict['mse'](denorm_pred_tensor, denorm_tg_tensor)
            mae_loss = criterion_dict['mae'](denorm_pred_tensor, denorm_tg_tensor)
            hub_loss = criterion_dict['hub'](denorm_pred_tensor, denorm_tg_tensor)

            # Accumulate metrics
            total_loss += hub_loss.item()
            total_mse += mse_loss.item()
            total_mae += mae_loss.item()
            predictions.append(denorm_pred)
            targets.append(denorm_tg)

    # Convert results to numpy for metrics
    predictions, targets = np.array(predictions), np.array(targets)

    return {
        'loss': total_loss / sample,
        'mse_loss': total_mse / sample,
        'mae': total_mae / sample,
        'r2': r2_score(targets, predictions),
        'predictions': predictions,
        'targets': targets,
    }