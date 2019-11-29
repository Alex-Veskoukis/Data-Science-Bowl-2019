def compare_shapes(x, y):
    print(x.shape[1] == y.shape[1])

def json_to_series(dataset):
    keys, values = zip(*[item for item in json.loads(dataset).items()])
    return pd.Series(values, index=keys)
