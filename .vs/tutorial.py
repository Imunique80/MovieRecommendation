

def feature_selection(df1, feature, target ,in_out, method='na'):
    fs_score =[]
    oe = OrdinalEncoder()
    X = (np.array(df1[feature])).reshape(-1,1)
    oe.fit(X)
    X_enc = oe.transform(X)

    y = np.array(df1[target]).reshape(-1,1)
    oe.fit(y)
    y_enc = oe.transform(y)

    if in_out == 'cat_cat':
        if method == 'chi2':
        fs = SelectKBest(score_func=chi2, k='all')
        else:
        fs = SelectKBest(score_func=mutual_info_classif, k='all')
        fs.fit(X_enc, y_enc)
        fs_score = fs.scores_
    elif in_out == 'num_num':
        fs = SelectKBest(score_func=f_regression, k='all')
        fs.fit(X, y.ravel())
        fs_score = fs.scores_
    elif in_out == 'num_cat':
        fs = SelectKBest(score_func=f_classif, k='all')
        fs.fit(X, y_enc)
        fs_score = fs.scores_
    elif in_out == 'cat_num':
        fs = SelectKBest(score_func=f_classif, k='all')
        fs.fit(X_enc, y.ravel())
        fs_score = fs.scores_
    else:
        fs_score=[]
    return fs_score



def get_col_type(df1,col_type):
    cols_types=df1.dtypes.reset_index()
    cols_types.columns=['col','type']
    cols_type = cols_types.apply(lambda x: x['col'] if x['type']==col_type else np.nan ,axis=1)
    return cols_type.dropna()

    def boxplot_by_col(df1,cat_cols,target):
    fig, ax = plt.subplots(len(cat_cols), 1, figsize=(25, 18))
    fig.subplots_adjust()
    t=0
    for var, subplot in zip(cat_cols, ax.flatten()):
        ax[t].set_xlabel(var,fontsize=18)
        sort_qtl_index = df1.groupby(var)[target].quantile(0.5).sort_values().index
        sort_qtl_values = df1.groupby(var)[target].quantile(0.5).sort_values()
        sns.boxplot(x=var, y=target, data=df1, ax=subplot,order=sort_qtl_index)
        sns.pointplot(x=sort_qtl_index,y= sort_qtl_values,ax=subplot,color='r')
        t+=1
    plt.tight_layout(pad=3)



def group_low_freq_cats(df1, col_name, threshold=0.01, name='others'):
    df1 = df1.copy()
    cat_freq = df1[col_name].value_counts()
    cat_low_freq = cat_freq[cat_freq/cat_freq.sum() <= threshold].index
    df1.loc[df1[col_name].isin(cat_low_freq),col_name]='others'
    return df1


def remove_incoherence(DataFrame,expression, replace_val, columns=[]):
    """ Problems Here"""

    if len(columns)==0:
        columns = DataFrame.columns

    DataFrame_aux=DataFrame.copy()

    if str(replace_val) == str(np.nan):
        DataFrame_aux=DataFrame.replace(expression, replace_val, regex=True)
        return DataFrame_aux
    else:
        for col in columns:
        i=0
        while True:
            DataFrame_aux[col]=DataFrame[col].str.replace(expression, replace_val, regex=True)
            num_matchs = len(DataFrame_aux[DataFrame_aux[col].str.contains(expression, na=False)])
            DataFrame = DataFrame_aux
            if num_matchs == 0:
                break
            if i == 100:
                DataFrame_aux =pd.DataFrame([])
                break
            i+=1
        return DataFrame_aux