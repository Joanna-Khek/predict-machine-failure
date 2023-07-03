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
from src.utils import pickle_save


def hyperparam_search(X: pd.DataFrame, y: pd.DataFrame, param_space: dict, model: str) -> float:
    """Hyperparameter tuning using hyperopt and MLflor

    Args:
        X (pd.DataFrame): Dataframe of features
        y (pd.DataFrame): Dataframe of target
        params (dict): dictionary of hyperparameters for the specified model
        model (str): name of model
    Returns:
        float: f1-score
    """

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", model)
            mlflow.log_params(params)
            
            k_fold = StratifiedKFold(n_splits=5, shuffle=True)
            fold_score = []

            for fold, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):
                # Get fold train and fold validation
                X_train, y_train = X.iloc[train_idx,:], y[train_idx]
                X_val, y_val = X.iloc[val_idx,:], y[val_idx]

                # Preprocess fold train and fold validation
                X_train_processed, X_val_processed = preprocess(X_train, 
                                                                X_val, 
                                                                feature_selection=True)

                clf = model_selector[model].set_params(**params)
                clf.fit(X_train_processed, y_train)
                y_pred = clf.predict(X_val_processed)

                score = f1_score(y_pred, y_val)
                fold_score.append(score)

            mlflow.log_metric("f1-score", np.mean(fold_score))

        return {'loss': -1.0 * np.mean(fold_score), 'status': STATUS_OK}
    
    rstate = np.random.default_rng(42)
    best_result = fmin(
        fn = objective, # function to optimize
        space = param_space, 
        algo = tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
        max_evals = 50, # maximum number of iterations
        trials = Trials(), # logging
        rstate = rstate # fixing random state for the reproducibility
    )

    return best_result

def process_params(best_result):
    """Change values type to int if it doesn't change the actual value.

    Args:
        best_result (dict): best params from hyperparameter tuning.
    """
    for key in best_result.keys():
        if int(best_result[key]) == best_result[key]:
            best_result[key] = int(best_result[key])
    return best_result

def best_performing_params(X: pd.DataFrame, y: pd.DataFrame, best_params: dict, model:str):
    """Use best performing parameters found in hyperparam tuning to get final f1-score

    Args:
        X (pd.DataFrame): features dataframe
        y (pd.DataFrame): target
        best_params (dict): best params based on hyperparam tuning
        model (str): name of selected name based on model_dispatcher
    """
    best_params = process_params(best_params)
    
    k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    fold_score = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):
        # Get fold train and fold validation
        X_train, y_train = X.iloc[train_idx,:], y[train_idx]
        X_val, y_val = X.iloc[val_idx,:], y[val_idx]

        # Preprocess fold train and fold validation
        X_train_processed, X_val_processed = preprocess(X_train, 
                                                        X_val, 
                                                        feature_selection=True)

        clf = model_selector[model].set_params(**best_params)
        clf.fit(X_train_processed, y_train)
        y_pred = clf.predict(X_val_processed)

        score = f1_score(y_pred, y_val)

        fold_score.append(score)


    print(f"F1-Score = {np.mean(fold_score)}")

    return clf

def run_search(X: pd.DataFrame, y: pd.DataFrame, model_name: str, model_params: dict):
    """Return F1 Score of selected model using best parameters found from hyperparam tuning

    Args:
        model_name (str): name of model in model_dispatcher
        model_params (dict): param search space for selected model
    """
    print(f"========{model_name}========")
    # Search for best param for selected model
    EXPERIMENT_NAME = f'{model_name}-hyperopt-cv'
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_result = hyperparam_search(X, y, model_params, model_name)
    clf = best_performing_params(X, y, best_result, model_name)
    print(best_result)

    # save model
    pickle_save(clf, "models/", model_name)

    return clf