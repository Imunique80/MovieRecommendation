
def mean(x):
    return sum(x)/len(x)


def removeNaN(df, col_name):
    data = df[col_name].to_numpy()

    
    for index, row in enumerate(data):
        if str(row) == "nan":
            data[index] = (str(float(0)))
    df[col_name] = data
    return df
