import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier

import mlflow
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin, partial
from hyperopt.pyll import scope

from src.preprocessing import preprocess
from src.model_dispatcher import model_selector
from src.hyperparam import hyperparam_search, best_performing_params
from src.utils import pickle_save


def train(X: pd.DataFrame, y: pd.DataFrame, k_fold, model, feature_selection=False):
    """Main training function

    Args:
        X (pd.DataFrame): Dataframe of features
        y (pd.DataFrame): Dataframe of targets
        k_fold : stratified K-fold indices
        model : model used for prediction
        feature_selection (bool, optional): Selection of features. Defaults to False.
    """
    fold_score = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):

        # Get fold train and fold validation
        X_train, y_train = X.iloc[train_idx,:], y[train_idx]
        X_val, y_val = X.iloc[val_idx,:], y[val_idx]

        # Preprocess fold train and fold validation
        X_train_processed, X_val_processed = preprocess(X_train, X_val, feature_selection)

        model = model
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_val_processed)

        score = f1_score(y_pred, y_val)
        fold_score.append(score)

        print(f"Fold = {fold+1}, F1 Score = {score}")
    print(f"Avg F1 Score = {np.mean(fold_score)}")
    return model



def training_baseline(X: pd.DataFrame, y: pd.DataFrame, feature_selection=False):
    """Function to train baseline results.

    Args:
        X (pd.DataFrame): Dataframe of features
        y (pd.DataFrame): Dataframe of targets
        feature_selection (bool, optional): Selection of features. Defaults to False.
    """

    k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    print(f"\n [Feature Selection == {feature_selection}]")
    for model_name in model_selector:
        print("===================")
        print(f"Model : {model_name}")
        print("===================")

        model = model_selector[model_name]
        train(X, y, k_fold, model, feature_selection)
     

def ensemble(X, y, estimators: list, voting='hard', feature_selection=True):
    """Function to train ensemble model

    Args:
        X (pd.DataFrame): Dataframe of features
        y (pd.DataFrame): Dataframe of targets
        estimators (list): list of models to ensemble
        voting (str, optional): voting type. Defaults to 'hard'.
        feature_selection (bool, optional): selection of features. Defaults to True.
    """
    print(f"\n Ensembling]")
    k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    model = VotingClassifier(estimators=estimators, voting = voting)
    ens_model = train(X, y, k_fold, model, feature_selection)
    
    # save model
    pickle_save(ens_model, path="models/", filename="ensemble")

    






   

