import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

class OrdinalEncoderWrapper:
    def __init__(self, ordinal_features, order_list):
        self.ordinal_features = ordinal_features
        self.encoder = None
        self.order_list = order_list

    def fit_transform(self, data):
        self.encoder = OrdinalEncoder(categories=self.order_list).set_output(transform="pandas")
        ordinal_encoded = self.encoder.fit_transform(data[self.ordinal_features])
        return ordinal_encoded


class OneHotEncoderWrapper:
    def __init__(self, one_hot_features):
        self.one_hot_features = one_hot_features
        self.encoder = None

    def fit_transform(self, data):
        self.encoder = OneHotEncoder(sparse=False).set_output(transform="pandas")
        one_hot_encoded = self.encoder.fit_transform(data[self.one_hot_features])
        return one_hot_encoded


class CustomEncoder:
    def __init__(self, feature_list, order_list):
        self.ordinal_encoder = OrdinalEncoderWrapper(feature_list, order_list)
        self.one_hot_encoder = OneHotEncoderWrapper(feature_list)

    def fit_transform(self, data):
        ordinal_encoded = self.ordinal_encoder.fit_transform(data)
        one_hot_encoded = self.one_hot_encoder.fit_transform(data)
        return ordinal_encoded, one_hot_encoded