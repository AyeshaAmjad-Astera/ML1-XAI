import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, file_path, target_column, test_size=0.2, random_state=42):
        cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        self.data = pd.read_csv(file_path, names = cols)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
        self.train_data = None
        self.test_data = None

        self.load_data()
        
    def load_data(self):
        self.features = self.data.drop(self.target_column, axis=1)
        self.target = self.data[self.target_column]
        
        self.train_data, self.test_data, train_target, test_target = train_test_split(
            self.features, self.target, test_size=self.test_size, random_state=self.random_state
        )
        
        self.train_data[self.target_column] = train_target
        self.test_data[self.target_column] = test_target
    
    def get_data(self):
        return self.features, self.target
    
    def get_df(self):
        return self.data
    
    def processed_data(self):
        pass
    
    def get_train_data(self):
        return self.train_data.drop(self.target_column, axis=1), self.data[self.target_column]
    
    def get_test_data(self):
        return self.test_data.drop(self.target_column, axis=1), self.data[self.target_column]