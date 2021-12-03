#! /usr/bin/env python



import os
import warnings
import pandas as pd
import statistics
import numpy as np
import seaborn as sns


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
from matplotlib        import pyplot as plt
from matplotlib.pyplot import figure


plt.style.use('ggplot')





def exclui_outliers(df1, col_name):
    Q1 = df1[col_name].quantile(.25)
    Q3 = df1[col_name].quantile(.75)
    IIQ =Q3 -Q1
    limite_inf = Q1 -1.5*IIQ
    limite_sup = Q3 +1.5*IIQ
    return df1[(df1[col_name]>=limite_inf) & (df1[col_name]<=limite_sup)]

def removeNaN(df, col_name):
    for index, row in enumerate(df[col_name].to_numpy()):
        if str(row) == "nan":
            # print(row)
        # if np.isnan(row):
            df[col_name][index] = str(float(0))
    return df

def group_low_freq_cats(df1, col_name, threshold=0.01, name='others'):
    df1 = df1.copy()
    cat_freq = df1[col_name].value_counts()
    cat_low_freq = cat_freq[cat_freq/cat_freq.sum() <= threshold].index
    df1.loc[df1[col_name].isin(cat_low_freq),col_name]='others'
    return df1

def val_couts_cols (df1,cols):
    """ Problems Here"""

    for x in cols:
        print('column: {0}, categories: {1}'.format(x,len(df1[x].value_counts())))
    print('Total samples: ' + str(len(df1)))




def main():
    print("Booting up the Movie Recommended System ...")

if __name__ == "__main__":
    main()
