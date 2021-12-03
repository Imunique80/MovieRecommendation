
def mean(x):
    return sum(x)/len(x)


def removeNaN(df, col_name):
    for index, row in enumerate(df[col_name].to_numpy()):
        if str(row) == "nan":
            # print(row)
        # if np.isnan(row):
            df[col_name][index] = str(float(0))
    return df

    