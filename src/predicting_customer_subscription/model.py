import pickle
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective="reg:squarederror")

    def train(self):
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return self.model

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path="model.pkl"):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, X):
        return self.model.predict(np.array(X))

if __name__ == "__main__":
    model = Model()
    model.train()
    model.save()
