from src.preprocessing import preprocess
from src.utils import pickle_load

def predict(X_train, df_test):
    X_train_processed, X_test_processed = preprocess(X_train, df_test, feature_selection=True)

    ens_model = pickle_load(filename="ensemble", path="models/")
    y_pred = ens_model.predict(X_test_processed)
    return y_pred