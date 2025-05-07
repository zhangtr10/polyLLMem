import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import numpy as np
## DATASET
class PolymerDataset(Dataset):
    def __init__(self, polymer_data, gpt_embedding_file, conf_embedding_file, unimol_model=None, normalizer=None,
                do_normalize = True):
        """
        Args:
           polymer_data: List of dictionaries with smiles, description, tg
           gpt_embedding_file: File path to pre-computed Llama embeddings 
           unimol_model: UniMol model for conformer embeddings
           normalizer: Instance of TgNormalizer for normalizing Tg values
        """
        # Filter data with valid Tg values
        self.data = [
            item for item in polymer_data
            if (item['metadata']['glass_transition_temp'] is not None and 
                item['metadata']['glass_transition_temp'] != '-' and
                item['metadata']['glass_transition_temp'] != '' )
        ]
        print(len(self.data))
     #and float(item['metadata']['glass_transition_temp'])<3 and float(item['metadata']['glass_transition_temp'])>0.1
        self.total_samples = len(polymer_data)
        self.failed_conformers = 0
        self.failed_embeddings = 0

        # Load pre-computed GPT embeddings
        with open(gpt_embedding_file, "rb") as fp:
            self.gpt_embeddings = pickle.load(fp)

        with open(conf_embedding_file, "rb") as fp2:
            self.conf_embeddings = pickle.load(fp2)

        # Normalize Tg values
        self.normalizer = normalizer
        if self.normalizer and do_normalize:
            tg_values = [float(item['metadata']['glass_transition_temp']) for item in self.data]
            self.normalizer.fit(np.array(tg_values))
        
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        item = self.data[idx]
        # Get pre-computed embeddings
        gpt_embed = self.gpt_embeddings[item['id']]
        conf_embed = self.conf_embeddings[item['id']]
        # Get and normalize Tg value
        tg_value = float(item['metadata']['glass_transition_temp'])
        #tg_value = math.log(tg_value)
        if self.normalizer:
            tg_value = self.normalizer.normalize(np.array([tg_value]))[0]

        return {
           'gpt_embed': torch.as_tensor(gpt_embed, dtype=torch.float32),
           'conf_embed': torch.as_tensor(conf_embed, dtype=torch.float32) if conf_embed is not None else None,
           'tg_value': torch.tensor(tg_value, dtype=torch.float32),
           'smiles': item['metadata']['polymer_smiles']  # Keep SMILES for reference
        }
       
def collate_fn(batch):
    """
    Custom collate to handle possible None conformer embeddings
    """
    # Filter out samples with None embeddings
    valid_batch = [b for b in batch if b['conf_embed'] is not None]
   
    if len(valid_batch) == 0:
        raise RuntimeError("No valid samples in batch")
       
    return {
       'gpt_embed': torch.stack([item['gpt_embed'] for item in valid_batch]),
       'conf_embed': torch.stack([item['conf_embed'] for item in valid_batch]),
       'tg_value': torch.stack([item['tg_value'] for item in valid_batch]),
       'smiles': [item['smiles'] for item in valid_batch]
    }


class TgNormalizer:
    def __init__(self, method='zscore', min_value=None, max_value=None, mean=None, std=None):
        self.method = method
        self.min_value = min_value
        self.max_value = max_value
        self.mean = mean
        self.std = std

    def fit(self, tg_values):
        if self.method == 'minmax':
            self.min_value = tg_values.min()
            self.max_value = tg_values.max()
        elif self.method == 'zscore':
            self.mean = tg_values.mean()
            self.std = tg_values.std()

    def normalize(self, tg_values):
        if self.method == 'minmax':
            return (tg_values - self.min_value) / (self.max_value - self.min_value)
        elif self.method == 'zscore':
            return (tg_values - self.mean) / self.std

    def denormalize(self, tg_values):
        if self.method == 'minmax':
            return tg_values * (self.max_value - self.min_value) + self.min_value
        elif self.method == 'zscore':
            return tg_values * self.std + self.mean

def split_data(polymer_data, train_ratio=0.8, val_ratio=0.2):
    """Split data into train, validation and test sets"""
    assert abs(train_ratio + val_ratio  - 1.0) < 1e-5
    
    total_size = len(polymer_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = polymer_data[:train_size]
    val_data = polymer_data[train_size:train_size+val_size]
    
    return train_data, val_data