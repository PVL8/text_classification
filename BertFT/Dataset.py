import io
import os
import json
import random
import tarfile
from abc import abstractmethod

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from transformers import BertTokenizer
from SomeDefs import SomeDefs

from sklearn.model_selection import train_test_split

#Стандартный класс датасета, наследуемого от класса Dataset модуля библиотеки torch. Как результат - экземляры датасета
#состоящие из тензоров текста, маски и класса после энкодера
class TextDataset(Dataset):

    def __init__(self, data, tokenizer, max_len=1024, seed=1234):
        
        self.data = data
        self.batch_size = 1
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        
        appeal_txt = self.data['txt'].iloc[idx]
        appeal_class = self.data['class'].iloc[idx]
        model_input = self.tokenizer.encode_plus(
            appeal_txt,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': appeal_txt,
            'input_ids': model_input['input_ids'].flatten(),#[:self.max_len-1]+model_input['input_ids'].flatten()[-1:],
            'attention_mask': model_input['attention_mask'].flatten(),#[:self.max_len-1]+model_input['input_ids'].flatten()[-1:],
            'target_class': torch.tensor(appeal_class, dtype=torch.long)
        }
        
def get_dataset(tokenizer, train=True):
    preps = SomeDefs()
    data = preps.prepare_dataset()
    x_train, x_test, y_train, y_test = train_test_split(data[data.columns[:-1]], data[data.columns[-1]], test_size=0.1, random_state=42)
    if train==True:
        x_train['class'] = y_train
        return TextDataset(x_train, tokenizer)
    else:
        x_test['class'] = y_test
        return TextDataset(x_test, tokenizer)