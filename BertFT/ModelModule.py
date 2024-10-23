import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from DataModule import AppsDataModule

#Основной класс с описанием модели, хода обучения и сохранения логов и параметров модели
class ClassificationModule(pl.LightningModule):
    def __init__(self, model_name='cointegrated/rubert-tiny2', n_classes=2840, model_save_path='/savings/bert.pt', n_warmup_steps=None, n_training_steps=None):
        super().__init__()
        
        self.model_name = model_name
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        self.model_save_path = model_save_path
        self.loss = nn.CrossEntropyLoss()
        self.training_step_outputs = []
        self.training_step_preds = []
        self.training_step_labels = []
        #self.n_warmup_steps = n_warmup_steps
        #self.n_training_steps = n_training_steps
        
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask)
        loss = self.loss(output.logits, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["target_class"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.training_step_outputs.append(loss)
        self.training_step_preds.append(outputs)
        self.training_step_labels.append(labels)
        return {"loss": loss, "predictions": outputs, "labels": labels}
        
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["target_class"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["target_class"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
        
    def predict_step(self, batch, batch_idx):
        return 0
        
    def on_train_epoch_end(self):
        predictions = []
        for out_predictions in self.training_step_preds[:-1]:
            predictions.append(torch.argmax(out_predictions.logits, dim=1))
        labels = torch.reshape(torch.stack(self.training_step_labels[:-1]), (-1, ))
        predictions = torch.reshape(torch.stack(predictions), (-1, ))
        
        avg_loss = torch.stack(self.training_step_outputs[:-1]).mean()
        avg_accuracy = torch.sum(predictions == labels)/len(labels)
        
        logs = {'avg_loss': avg_loss, 'avg_accuracy': avg_accuracy}
        tensorboard_logs = {'train/avg_loss': avg_loss, 'train/avg_accuracy': avg_accuracy}
        results = {'progress_bar': logs, 'log': tensorboard_logs}
        
        print(tensorboard_logs)
        
        self.training_step_outputs.clear()
        self.training_step_preds.clear()
        self.training_step_labels.clear()
        
        return results
          
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        #scheduler = get_linear_schedule_with_warmup(
        #  optimizer,
        #  num_warmup_steps=self.n_warmup_steps,
        #  num_training_steps=self.n_training_steps
        #)
        return dict(
          optimizer=optimizer,
          #lr_scheduler=dict(
          #  scheduler=scheduler,
          #  interval='step'
          #)
        )
        
def run_train(conf, model_path='cointegrated/rubert-tiny2'):

    pl.seed_everything(1234)
    
    checkpoint_callback = ModelCheckpoint(**conf.model_checkpoint)
    
    logger = TensorBoardLogger("lightning_logs", name="app_classification")
    
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
    
    trainer = pl.Trainer(
      logger=logger,
      callbacks=[checkpoint_callback, early_stopping_callback],
      max_epochs=10,
      log_every_n_steps=30
    )
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    data_module = AppsDataModule(tokenizer=tokenizer)
    model = ClassificationModule()
    
    ckpt_path = getattr(conf.common, 'ckpt_path', None)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)