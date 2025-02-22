from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocess:
    def __init__(self, data, target, test_size=0.2):
        self.data = data
        self.target = target
        self.test_size = test_size


    def encode(self, X):
        # One-hot encoding of the categorical columns
        cat_cols = X.select_dtypes(include=['object']).columns
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = pd.DataFrame(enc.fit_transform(X[cat_cols]), index=X.index)
        X_cat.columns = enc.get_feature_names_out(cat_cols)
        X = pd.concat([X.drop(cat_cols, axis=1), X_cat], axis=1)
        return X

    def split_data(self, encode=True):
        # Split the data
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target].map({'yes': 1, 'no': 0})

        # Encode the data if required
        if encode:
            X = self.encode(X)

        # Train-test split
        X_train, X_test, y_train, y_test = (
            train_test_split(X, y, test_size=self.test_size,
                             stratify=y,
                             random_state=42))

        return X_train, X_test, y_train, y_test