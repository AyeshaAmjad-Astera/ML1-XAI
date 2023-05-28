import sys
sys.path.append("../")
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from utils.encoding import Encoded_Features

class DataLoader:
    
    """
    DataLoader class to load data.

    load_data function splits orginal data, ordinal encoded data and One-Hot encoded data into train and test.

    """

    def __init__(self, file_path, target_column, test_size=0.2, random_state=42, clean_data=False):
        
        self.data = pd.read_csv(file_path)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.clean_data = clean_data
        
        self.train_data = None
        self.test_data = None
        
        self.train_data_ord = None
        self.test_data_ord = None
        
        self.train_data_ohe = None
        self.test_data_ohe = None

        self.load_data()
        
    def load_data(self):
        if(self.clean_data):
            self.processed_data()
        self.features = self.data.drop(self.target_column, axis=1)
        self.target = self.data[self.target_column]

        self.encoded_data = Encoded_Features(self.features.select_dtypes(include=['object']).columns.to_list(), self.features.select_dtypes(include=['category']).columns.to_list())
        self.encoded_data = self.encoded_data.fit_transform(self.features)

        self.train_data, self.test_data, train_target, test_target = train_test_split(
            self.features, self.target, test_size=self.test_size, random_state=self.random_state, stratify=self.target
        )
        
        self.train_data[self.target_column] = train_target
        self.test_data[self.target_column] = test_target

        self.train_data_enc, self.test_data_enc, train_target, test_target = train_test_split(
            self.encoded_data, self.target, test_size=self.test_size, random_state=self.random_state, stratify=self.target
        )
        
        self.train_data_enc[self.target_column] = train_target
        self.test_data_enc[self.target_column] = test_target
        
    def get_data(self):
        return self.features, self.target
    
    def get_df(self):
        return self.data
    
    def processed_data(self):
        self.data.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True, axis=1)
        self.data['NumOfProducts'] = self.data['NumOfProducts'].astype('category', copy=False)
        self.data['HasCrCard'] = self.data['HasCrCard'].astype('category', copy=False)
        self.data['IsActiveMember'] = self.data['IsActiveMember'].astype('category', copy=False)
        self.data['Tenure'] = self.data['Tenure'].astype('category', copy=False)
        z_score = np.abs(stats.zscore(self.data['Age']))
        self.data = self.data[(z_score < 3)]
    
    def get_train_data(self):
        return self.train_data.drop(self.target_column, axis=1), self.train_data[self.target_column]
    
    def get_test_data(self):
        return self.test_data.drop(self.target_column, axis=1), self.test_data[self.target_column]
    
    def get_data_enc(self):
        return self.encoded_data, self.target
    
    def get_train_data_enc(self):
        return self.train_data_enc.drop(self.target_column, axis=1), self.train_data_enc[self.target_column]
    
    def get_test_data_enc(self):
        return self.test_data_enc.drop(self.target_column, axis=1), self.test_data_enc[self.target_column]