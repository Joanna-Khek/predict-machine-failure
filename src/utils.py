from pathlib import Path
import pandas as pd
import pickle

from src.config import config


def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(Path(config['raw_data_path'], config[filename]))
    df = (df
          .rename(columns=config['columns'])
          .drop(config['drop_columns'], axis=1)
    )
    
    return df

def pickle_save(object, path, filename: str):
    filename = filename + ".pickle"
    final_path = Path(path, filename)
    with open(final_path, 'wb') as handle:
      pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(filename: str, path):
    filename = filename + ".pickle"
    final_path = Path(path, filename)
    with open(final_path, 'rb') as handle:
      b = pickle.load(handle)
      return b
    
def nulls_summary_table(df):
    """
    Returns a summary table showing null value counts and percentage
    
    Parameters:
    df (DataFrame): Dataframe to check
    
    Returns:
    null_values (DataFrame)
    """
    null_values = pd.DataFrame(df.isnull().sum())
    null_values[1] = null_values[0]/len(df)
    null_values.columns = ['null_count','null_pct']
    return null_values