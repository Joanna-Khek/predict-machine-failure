import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from src.config import config

class FrequencyEncoder:
    """Encodes categorical variables using frequency
    """
    def fit(self, df_train, column):
        self.df_train = df_train
        self.column = column

    def transform(self, df_test, column):
        
        # Obtain the frequency using train set
        frequency_encoded = self.df_train.groupby([self.column]).size()

        df_frequency_encoded = (frequency_encoded
                                .reset_index()
                                .rename(columns={0:'Count'}))
        
        col_name = column + '_freq'
        
        # Handles cases where categories not found in train set but in test
        # Assign Count = 0 for categories not found in train set
        df_test_col_values = pd.Series(df_test[[column]].squeeze())
        df_train_col_values = self.df_train[[column]].squeeze()

        missing_category = pd.Series(df_test[~df_test_col_values.isin(df_train_col_values)][[column]].squeeze())
        df_missing_category = (missing_category
                             .reset_index()
                             .drop("index", axis=1)
                             .rename(columns={0: 'ProductID'})
                             .assign(Count = 0))
        
        full_frequency_encoded = pd.concat([df_frequency_encoded,
                                            df_missing_category], axis=0, ignore_index=True)
        full_frequency_encoded = (full_frequency_encoded
                                  .drop_duplicates()
                                  .set_index(column)
                                  .squeeze())

        # Map the frequency count
        df_test = df_test.copy()
        df_test.loc[:,col_name] = (df_test_col_values
                                   .apply(lambda x: full_frequency_encoded[x]))
        
        df_test = (df_test
                   .assign(ProductID_freq=lambda df_: df_.ProductID_freq.astype(int))
        )

        return df_test
    
def get_encoder_inst(feature_col: pd.Series) -> OneHotEncoder:
    """One hot encoding
    Args: 
        feature_col (pd.Series): Feature from the dataframe
    
    Returns:
        an instance of sklearn OneHotEncoder
    """
    assert isinstance(feature_col, pd.Series)
    feature_vec = feature_col.sort_values().values.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(feature_vec) 
    return enc 

def get_one_hot_enc(feature_col: pd.Series, enc: OneHotEncoder)-> pd.DataFrame:
    """maps an unseen column feature using one-hot-encoding previously fit against training data 
    Args:
        feature_col (pd.Series): Feature from the dataframe
        enc (OneHotEncoder): The encoding instance

    Returns:
      pd.DataFrame of newly one-hot-encoded feature
    """
    assert isinstance(feature_col, pd.Series)
    assert isinstance(enc, OneHotEncoder)
    unseen_vec = feature_col.values.reshape(-1, 1)
    encoded_vec = enc.transform(unseen_vec).toarray()
    encoded_df = pd.DataFrame(encoded_vec)
    encoded_df = encoded_df.astype(int)
    encoded_df.columns = enc.categories_[0]
            
    return encoded_df

def convert_type(df):
    df.ProductID = df.ProductID.astype(object)
    df.Type = df.Type.astype(object)
    df.AirTemperature = df.AirTemperature.astype(float)
    df.ProcessTemperature = df.ProcessTemperature.astype(float)
    df.RotationalSpeed = df.RotationalSpeed.astype(int)
    df.Torque = df.Torque.astype(float)
    df.ToolWear = df.ToolWear.astype(int)
    df.TWF = df.TWF.astype(int)
    df.HDF = df.HDF.astype(int)
    df.PWF = df.PWF.astype(int)
    df.OSF = df.OSF.astype(int)
    df.RNF = df.RNF.astype(int)

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

    # Ensure that df_test is a dataframe
    cols = ["ProductID", "Type", "AirTemperature", "ProcessTemperature", "RotationalSpeed",
            "Torque", "ToolWear", "TWF", "HDF", "PWF", "OSF", "RNF"]
    
    df_test = pd.DataFrame(df_test, columns=cols)

    convert_type(df_train)
    convert_type(df_test)

    # 1. Frequency Encoder
    fe = FrequencyEncoder()
    fe.fit(df_train, column='ProductID')
    test_fe = fe.transform(df_test, column='ProductID')
    train_fe = fe.transform(df_train, column='ProductID')

    # 2. One Hot Encoding
    enc = get_encoder_inst(train_fe.Type)
    test_ohe = get_one_hot_enc(test_fe.Type, enc)
    train_ohe = get_one_hot_enc(train_fe.Type, enc)

    train_encoded = (pd.concat([train_fe.reset_index(drop=True), train_ohe], axis=1)
                         .drop(["ProductID", "Type"], axis=1))
    
    test_encoded = (pd.concat([test_fe.reset_index(drop=True), test_ohe], axis=1)
                        .drop(["ProductID", "Type"], axis=1))
    
    # 3. Feature Selection
    if feature_selection == True:
        train_encoded = train_encoded.drop(config['feature_selection_drop_columns'], axis=1)
        test_encoded = test_encoded.drop(config['feature_selection_drop_columns'], axis=1)

    return train_encoded, test_encoded

# def scaling(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:

#     std_scale = StandardScaler().fit(X_train)
#     X_train_scaled = pd.DataFrame(std_scale.transform(X_train), columns = X_train.columns)
#     X_test_scaled = pd.DataFrame(std_scale.transform(X_test), columns = X_train.columns)

#     return X_train_scaled, X_test_scaled

