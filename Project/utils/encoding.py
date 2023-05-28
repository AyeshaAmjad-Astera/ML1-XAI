import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class OneHotEncoderWrapper:
    def __init__(self, one_hot_features):
        self.one_hot_features = one_hot_features
        self.encoder = None

    def fit_transform(self, data):
        self.encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        one_hot_encoded = self.encoder.fit_transform(data[self.one_hot_features])
        return one_hot_encoded


class CatEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns, dtype):
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = X[self.columns].astype(self.dtype)
        return X

# class CatEncoderWrapper:
#     def __init__(self, cat_features):
#         self.cat_features = cat_features
#     def fit_transform(self, data):
#         for i in self.cat_features:
#             data = data.apply(lambda x: x.astype('int64'))
#         return data
class Encoded_Features:
    def __init__(self, object_features, cat_features):
        self.object_features = object_features
        self.cat_features = cat_features
        self.one_hot_encoder = OneHotEncoderWrapper(self.object_features)
        self.cat_encoder = CatEncoderWrapper(self.cat_features, 'int64')
    def fit_transform(self, data):
        df_ohe = data[self.object_features]
        df_cat = data[self.cat_features]
        df = data[data.columns.difference(self.object_features + self.cat_features)]
        one_hot_encoded = self.one_hot_encoder.fit_transform(df_ohe)
        cat_encoded = self.cat_encoder.fit_transform(df_cat)
        return pd.concat([df, one_hot_encoded, cat_encoded], axis=1)