import pandas as pd

from category_encoders.count import CountEncoder
from category_encoders.one_hot import OneHotEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.config import config



def convert_type(df):
    df_update = (df
          .assign(
                  AirTemperature = lambda df_: df_.AirTemperature.astype(float),
                  ProcessTemperature = lambda df_: df_.ProcessTemperature.astype(float),
                  RotationalSpeed = lambda df_: df_.RotationalSpeed.astype(int),
                  Torque = lambda df_: df_.Torque.astype(float),
                  ToolWear = lambda df_: df_.ToolWear.astype(int),
                  TWF = lambda df_: df_.TWF.astype(int),
                  HDF = lambda df_: df_.HDF.astype(int),
                  PWF = lambda df_: df_.PWF.astype(int),
                  OSF = lambda df_: df_.OSF.astype(int),
                  RNF = lambda df_: df_.RNF.astype(int)
          )
    )

    return df_update

def preprocess(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_selection: bool=True) -> pd.DataFrame:
    """Performs One Hot Encoding and Frequency Encoder for categorical columns.
       Removes features through feature selection process

    Args:
        df_train (pd.DataFrame): train set
        df_test (pd.DataFrame): test set
        feature_selection (bool): If feature selection = True, use feature selection.

    Returns:
        pd.DataFrame: Encoded data for both train set and test set
    """
    
    cols = ['ProductID', 'Type', 'AirTemperature', 'ProcessTemperature', 'RotationalSpeed',
            'Torque', 'ToolWear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    df_test = pd.DataFrame(df_test, columns=cols)

    
    # Pipeline
    product_count_encoder = CountEncoder(cols='ProductID', handle_unknown=0)
    one_hot_encoder = OneHotEncoder(cols='Type', handle_unknown='Unknown', use_cat_names=True)
    scaler = StandardScaler()

    pipeline = make_pipeline(product_count_encoder, one_hot_encoder)
    
    pipeline.fit(df_train)
    train_encoded = pd.DataFrame(pipeline.transform(df_train), columns=pipeline.get_feature_names_out())
    test_encoded = pd.DataFrame(pipeline.transform(df_test), columns=pipeline.get_feature_names_out())

    #Convert types
    train_encoded = convert_type(train_encoded)
    test_encoded = convert_type(test_encoded)
   
    # Feature Selection
    if feature_selection == True:
        train_encoded = train_encoded.drop(config['feature_selection_drop_columns'], axis=1)
        test_encoded = test_encoded.drop(config['feature_selection_drop_columns'], axis=1)


    return train_encoded, test_encoded


