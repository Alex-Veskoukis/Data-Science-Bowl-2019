import pandas as pd
import json
def json_to_series(dataset):
    """
    This function gets a pandas Series that is structured as JSON,
    reads its keys and values
    saves a pd.Series with keys the column names and values the data.
    To be used along with pd.concat
    """
    keys, values = zip(*[item for item in json.loads(dataset).items()])
    return pd.Series(values, index=keys)

train = pd.read_csv('./train.csv')

trainWithJson = pd.concat([train, train['event_data'].apply(json_to_series)], axis=1)
print(trainWithJson.describe())