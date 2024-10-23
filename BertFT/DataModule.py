import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from Dataset import TextDataset, get_dataset

RANDOM_SEED = 1234

#Стандартный класс лайтнинга с инициализацией датасета и даталоадеров
class AppsDataModule(pl.LightningDataModule):
  def __init__(self, tokenizer, batch_size=400, max_token_len=1024):
    super().__init__()
    self.batch_size = batch_size
    self.tokenizer = tokenizer
    self.train_dataset = get_dataset(self.tokenizer, train=True)
    self.test_dataset = get_dataset(self.tokenizer, train=False)
    
  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=0
    )
    
  def val_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=0
    )
    
  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=0
    )
