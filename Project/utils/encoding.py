import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

class OneHotEncoderWrapper:
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.column_transformer = None

    def fit_transform(self, X):
        self.column_transformer = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(sparse_output=False), self.categorical_features)],
            remainder='passthrough', verbose_feature_names_out=False
        ).set_output(transform="pandas")
        transformed_data = self.column_transformer.fit_transform(X)
        return transformed_data


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
class Encoded_Features:
    def __init__(self, object_features, cat_features):
        self.object_features = object_features
        self.cat_features = cat_features
        self.one_hot_encoder = OneHotEncoderWrapper(self.object_features)
        self.cat_encoder = CatEncoderWrapper(self.cat_features, 'int64')
    def fit_transform(self, data):
        one_hot_encoded = self.one_hot_encoder.fit_transform(data)
        cat_encoded = self.cat_encoder.transform(one_hot_encoded)
        return cat_encoded