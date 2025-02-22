import pickle
import numpy as np
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt
from predicting_customer_subscription.utils import logger
import pandas as pd

log = logger()

class Model:
    def __init__(self, model_type='xgboost', n_splits=5):
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(objective="binary:logistic",
                                           seed=42, eval_metric='logloss', n_jobs=-1)
            self.param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5]
            }
        elif model_type == 'catboost':
            self.model = cb.CatBoostClassifier(verbose=1, loss_function='Logloss', thread_count=-1)
            self.param_grid = {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 4, 5]
            }
        else:
            raise ValueError("Unsupported model type. Choose 'xgboost' or 'catboost'.")
        self.model_type = model_type
        self.n_splits = n_splits

    def train(self, X, y, cat_features=None):
        #keep feature names for feature importance
        self.feature_names = X.columns

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid,
                                   cv=skf, scoring='neg_log_loss',
                                   verbose=1, refit=True)
        if self.model_type == 'catboost' and cat_features is not None:
            grid_search.fit(X, y, cat_features=cat_features)
        else:
            grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")

        return self.model

    def save(self, path="model.pkl"):
        log.info(f"Saving model to {path}")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path="model.pkl"):
        log.info(f"Loading model from {path}")
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'],
                             columns=['Predicted Negative', 'Predicted Positive'])
        print("Confusion Matrix:")
        print(cm_df)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        print(f"PR AUC: {pr_auc}")

        plt.figure()
        plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="best")
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc}")

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="best")
        plt.show()

    def feature_importance(self):

        if self.model_type == 'xgboost':
            gain_importance = self.model.get_booster().get_score(importance_type='gain')
            importance = [gain_importance.get(f, 0) for f in self.feature_names]
        elif self.model_type == 'catboost':
            importance = self.model.get_feature_importance()
        else:
            raise ValueError("Unsupported model type. Choose 'xgboost' or 'catboost'.")

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        return importance_df



# if __name__ == "__main__":
