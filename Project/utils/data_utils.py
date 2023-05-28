import sys
sys.path.append("../")
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.encoding import CustomEncoder

class DataLoader:
    
    """
    DataLoader class to load data.

    load_data function splits orginal data, ordinal encoded data and One-Hot encoded data into train and test.

    """

    def __init__(self, file_path, target_column, test_size=0.2, random_state=42):
        cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        
        buying_order = ['low', 'med', 'high', 'vhigh']
        maint_order = ['low', 'med', 'high', 'vhigh']
        doors_order = ['2', '3', '4', '5more']
        persons_order = ['2', '4', 'more']
        lug_boot_order = ['small', 'med', 'big']
        safety_order = ['low', 'med', 'high']

        self.order_list = [buying_order, maint_order, doors_order, persons_order, lug_boot_order, safety_order]
        
        self.data = pd.read_csv(file_path, names = cols)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
        self.train_data = None
        self.test_data = None
        
        self.train_data_ord = None
        self.test_data_ord = None
        
        self.train_data_ohe = None
        self.test_data_ohe = None

        self.load_data()
        
    def load_data(self):
        self.features = self.data.drop(self.target_column, axis=1)
        self.target = self.data[self.target_column]
        
        self.X_ord, self.X_ohe = CustomEncoder(self.features.columns, order_list=self.order_list).fit_transform(self.features)
        
        self.train_data, self.test_data, train_target, test_target = train_test_split(
            self.features, self.target, test_size=self.test_size, random_state=self.random_state, stratify=self.target
        )
        
        self.train_data[self.target_column] = train_target
        self.test_data[self.target_column] = test_target
        
        
        self.train_data_ord, self.test_data_ord, train_target, test_target = train_test_split(
            self.X_ord, self.target, test_size=self.test_size, random_state=self.random_state, stratify=self.target
        )
        
        self.train_data_ord[self.target_column] = train_target
        self.test_data_ord[self.target_column] = test_target
        
        
        self.train_data_ohe, self.test_data_ohe, train_target, test_target = train_test_split(
            self.X_ohe, self.target, test_size=self.test_size, random_state=self.random_state, stratify=self.target
        )
        
        self.train_data_ohe[self.target_column] = train_target
        self.test_data_ohe[self.target_column] = test_target
    
    def get_data(self):
        return self.features, self.target
    
    def get_df(self):
        return self.data
    
    def processed_data(self):
        pass
    
    def get_train_data(self):
        return self.train_data.drop(self.target_column, axis=1), self.train_data[self.target_column]
    
    def get_test_data(self):
        return self.test_data.drop(self.target_column, axis=1), self.test_data[self.target_column]
    
    def get_data_ord(self):
        return self.X_ord, self.target
    
    def get_data_ohe(self):
        return self.X_ohe, self.target
    
    def get_train_data_ord(self):
        return self.train_data_ord.drop(self.target_column, axis=1), self.train_data_ord[self.target_column]
    
    def get_test_data_ord(self):
        return self.test_data_ord.drop(self.target_column, axis=1), self.test_data_ord[self.target_column]
    
    def get_train_data_ohe(self):
        return self.train_data_ohe.drop(self.target_column, axis=1), self.train_data_ohe[self.target_column]
    
    def get_test_data_ohe(self):
        return self.test_data_ohe.drop(self.target_column, axis=1), self.test_data_ohe[self.target_column]