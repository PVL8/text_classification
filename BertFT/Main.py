import io
import os
import json
import random
import tarfile

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from SomeDefs import SomeDefs

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from transformers import BertForSequenceClassification

from omegaconf import OmegaConf
from ModelModule import run_train


conf = OmegaConf.load('configus.yml')
print("Запуск обучения: ")
run_train(conf)


