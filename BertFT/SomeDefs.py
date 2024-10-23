import io
import os
import json
import random
import tarfile

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

#Класс содержит функции для предобработки датасета, состоящего из пар Текст-Класс
class SomeDefs:

    def __init__(self):
        self.data_path = '****'
        self.classes_path = '****/Классы.xlsx'
        
        
    def prepare_dataset(self):
    
        files = os.listdir(self.data_path)
        classes = pd.read_excel(self.classes_path)
    
        main_df = pd.read_csv(self.data_path+files[0])
        for file in tqdm(files[1:]):
            buf_df = pd.read_csv(self.data_path+file)
            main_df = pd.concat([main_df, buf_df], ignore_index=True, axis=0)
    
        main_df.dropna(ignore_index=True, inplace=True)
        main_df['txt'] = main_df['txt'].apply(lambda x: x.split('Содержание обращения: \n')[1] if 'Содержание обращения: \n' in x else x)
        main_df['txt'] = main_df['txt'].apply(lambda x: x.split('Приложение')[0] if 'Приложение' in x else x)
        main_df['txt'] = main_df['txt'].apply(lambda x: x.replace('\n', ''))
        main_df['txt'] = main_df['txt'].apply(lambda x: x.strip('/'))
        main_df['class'] = main_df['code'].apply(lambda x: classes[classes['Код']==x]['Класс'].values[0] if x in list(classes['Код']) else 'Класс не определен')
        main_df = main_df[main_df['class']!='Класс не определен']
        
        
        return main_df