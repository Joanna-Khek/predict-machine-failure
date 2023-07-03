from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb

from hyperopt import hp
from hyperopt.pyll import scope

model_selector = {
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'gradient_boosting': GradientBoostingClassifier(),
    'adaptive_boosting': AdaBoostClassifier(),
    'xgboost': xgb.XGBClassifier(),
    'light_gradient_boosting': LGBMClassifier()
}

rf_params={'n_estimators': scope.int(hp.quniform('n_estimators', 100, 500, 50)),
           'max_depth': scope.int(hp.quniform('max_depth', 2, 6, 1)),
           'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 2, 20, 1)),
           'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1))}

gb_params={'learning_rate': hp.quniform('learning_rate', 0.1, 1, 0.001),
           'n_estimators': scope.int(hp.quniform('n_estimators', 2, 6, 1)),
           'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 2, 10, 1)),
           'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 5, 1)),
           'max_depth': scope.int(hp.quniform('max_depth', 3, 20, 1))}

ada_params={'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 1)),
            'learning_rate': hp.quniform('learning_rate', 0.1, 1, 0.001)}

xg_params={'eta': hp.quniform('eta', 0.1, 1.0, 0.001),
           'max_depth': scope.int(hp.quniform('max_depth', 3, 20, 1)),
           'colsample_bytree': hp.quniform('colsample_bytree', 0, 1, 00.1)}

lgbm_params={'max_depth': scope.int(hp.quniform('max_depth', 3, 20, 1)),
            'learning_rate': hp.quniform('learning_rate', 0.1, 1, 0.001),
            'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 10)),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 00.1)}