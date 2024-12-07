import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import pickle
import gzip
import json, os
np.set_printoptions(legacy='1.25')
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score, f1_score, confusion_matrix, precision_score)

class Lab01:
    def __init__(self) -> None:
        self.files_path = 'files/'
        self.columnas_categoricas = ['SEX','EDUCATION','MARRIAGE']
        self.param_grid = {
            'randomforest__n_estimators': [180,175],
            'randomforest__max_features': ['sqrt',23],
            'randomforest__min_samples_split': [3,5,10],
            'randomforest__min_samples_leaf': [2,4],
            'randomforest__bootstrap': [True],
            'randomforest__max_depth': [None]
        }

    def main(self):
        df_train = self.read_dataset('input/train_data.csv.zip')
        df_test = self.read_dataset('input/test_data.csv.zip')
        df_train = self.clean_dataset(df_train)
        df_test = self.clean_dataset(df_test)
        X_train,  y_train = self.train_test_split(df_train)
        X_test,  y_test = self.train_test_split(df_test)
        pipeline = self.make_pipeline(RandomForestClassifier(random_state=2024, verbose=1, n_jobs=-1))
        estimator = self.make_grid_search(pipeline, 'balanced_accuracy', StratifiedKFold(n_splits=10,shuffle=False))
        estimator = estimator.fit(X_train, y_train)
        estimator = self.save_model_if_best(estimator, X_train, y_train)
        y_train_pred = estimator.predict(X_train)
        y_test_pred = estimator.predict(X_test) 
        metrics_train = self.eval_metrics('train', y_train, y_train_pred)
        metrics_test = self.eval_metrics('test', y_test, y_test_pred)
        cm_train = self.eval_confusion_matrix('train', y_train, y_train_pred)
        cm_test = self.eval_confusion_matrix('test', y_test, y_test_pred)
        self.save_metrics(metrics_train, metrics_test, cm_train, cm_test)



    def read_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(self.files_path + path)
        return df
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={'default payment next month': 'default'})
        df.drop('ID', axis=1, inplace=True)
        df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x>4 else x)
        df = df.query('MARRIAGE != 0 and EDUCATION != 0')
        return df
    
    def train_test_split(self, df):
        return df.drop('default', axis=1), df['default']
    
    def make_pipeline(self, estimator):
        transformer = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(dtype='int'), self.columnas_categoricas)
            ],
            remainder='passthrough'
        )

        pipeline = Pipeline(
            steps=[
                ('transformer', transformer),
                ('randomforest', estimator)
            ]
        )
        return pipeline
        
    def make_grid_search(self, estimator, scoring, cv=10):
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=self.param_grid,
            cv=StratifiedKFold(n_splits=10,shuffle=False),
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        return grid_search

    def save_model_if_best(self, estimator, X, y):
        best_estimator = self.load_model()
        saved_accuracy = balanced_accuracy_score(y, best_estimator.predict(X))
        current_accuracy = balanced_accuracy_score(y, estimator.predict(X))

        if current_accuracy > saved_accuracy:
            with gzip.open(self.files_path + 'models/model.pkl.gz', 'wb') as file:
                pickle.dump(estimator, file)
        else:
            estimator = best_estimator
        return estimator


    def load_model(self):
        with gzip.open(self.files_path + "models/model.pkl.gz", "rb") as file:
            estimator = pickle.load(file)
        return estimator
    
    def eval_metrics(self, dataset,y_true, y_pred):
        accuracy = precision_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return {"type": "metrics","dataset": dataset, "precision": accuracy, "balanced_accuracy": balanced_accuracy, "recall": recall, "f1_score": f1} 
    
    def eval_confusion_matrix(self,dataset,y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_train_dict = {
            'type': 'cm_matrix',
            'dataset': dataset,
            'true_0': {
                'predicted_0': int(cm[0,0]),
                'predicted_1': int(cm[0,1])
            },
            'true_1': {
                'predicted_0': int(cm[1,0]),
                'predicted_1': int(cm[1,1])
            }
        }
        return cm_train_dict
    
    def save_metrics(self, metrics_train, metrics_test, cm_train, cm_test):
        os.remove(self.files_path + 'output/metrics.json')
        with open(self.files_path + 'output/metrics.json', mode='w') as file:
            file.write(json.dumps(metrics_train)+"\n")
            file.write(json.dumps(metrics_test)+"\n")
            file.write(json.dumps(cm_train)+"\n")
            file.write(json.dumps(cm_test)+"\n")


if __name__=='__main__':
    obj = Lab01()
    obj.main()