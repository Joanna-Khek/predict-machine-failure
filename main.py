from src.get_data import download_dataset
from src.config import config
from src.utils import read_data, pickle_load
from src.train import training_baseline, ensemble
from src.hyperparam import run_search
from src.model_dispatcher import rf_params, gb_params, ada_params, xg_params, lgbm_params
from src.predict import predict

if __name__ == "__main__":
    
    # download data
    #raw_data_path = config['raw_data_path']
    #download_dataset(raw_data_path)

    # Read data
    df_train = read_data("train_data")
    df_test = read_data("test_data")

    # Train data
    X = df_train.drop("MachineFailure", axis=1)
    y = df_train.MachineFailure

    # Baseline results without feature selection
    #training_baseline(X, y, feature_selection=False)

    # Baseline results with feature selection
    #training_baseline(X, y, feature_selection=True)

    # Find best hyperparams and train using those best params
    #RF = run_search(X, y, 'random_forest', rf_params)
    #GB = run_search(X, y, 'gradient_boosting', gb_params)
    #ADA = run_search(X, y, 'adaptive_boosting', ada_params)
    XGB = run_search(X, y, 'xgboost', xg_params)
    #LGBM = run_search(X, y, 'light_gradient_boosting', lgbm_params)

    # Load model
    #RF = pickle_load(filename="random_forest", path="models/")
    #GB = pickle_load(filename="gradient_boosting", path="models/")
    #ADA = pickle_load(filename="adaptive_boosting", path="models/")
    #XGB = pickle_load(filename="xgboost", path="models/")
    #LGBM = pickle_load(filename="light_gradient_boosting", path="models/")

    # Ensemble
    #estimator = [('RF', RF), ('GB', GB), ('ADA', ADA), ('XGB', XGB), ('LGBM', LGBM)]
    #ensemble(X, y, estimator, voting='hard', feature_selection=True)


   


    
