import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

def train(X,y):
    """KNeighborsClassifier"""
    model = KNeighborsClassifier(n_neighbors=3)
    return model.fit(X, y)


class Eval:
  def cross_val(model, XTests, yTests):
      yPreds = model.predict(XTests)
      results = cross_val_score(model, XTests, yTests, cv=5)
      return results



def generateFigure(model, XTests, yTests):
    yPreds = model.predict(XTests)

    X_Axis = [_ for _ in range(len(yPreds))]
    plt.plot(X_Axis, yPreds)
    plt.ylabel('Y')
    # plt.axis([0, 500, 0, 1.2])
    plt.savefig('results.png')

      # for idx,_ in enumerate(yPreds):
      #   yTest = yTests[idx]
      #   yPred = yPreds[idx]
      #   print(idx, yPred, "---",yTest)






def test():
    X, y = load_iris(return_X_y=True)
    print (X.shape)
    print (y.shape)

  # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.7, random_state=42)
  # knn = trainCluster(X,y)
  # modelEval(knn, X_test, y_test)

  # def runClustering():
  #     """ """
  #     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
  #     knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
  #     knn.fit(X_train, y_train)
  #     y_pred = knn.predict(X_test)
  #
  #     cat_cols = get_col_type(df1_fs, 'object')
  #     # val_couts_cols(df1_fs,cat_cols)
      # sns.scatterplot(
      #   #         x='mean gross',
      #   #         y='mean budget',
      #   #         hue='4500000',
      #   #         data=X_test.join(y_test, how='outer'))




if __name__ == "__main__":
    df_clean = clean_data(data = df1)
    y = df_clean[y_col[0]].to_numpy()
    X = df_clean[X_col].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    knn = trainCluster(X,y)
    modelEval(knn, X_test, y_test)



  # X =s

  # df1_fs = df1.drop(['country','votes','runtime','released'], axis=1)
  # print (f"df1: {df1_fs}")
  # print (f"df1 info: {df1_fs.info}")
  # df1_fs = removeNaN(df1_fs, "budget")
  # df1_fs = removeNaN(df1_fs, "gross")
  # print ()
  # print (df1_fs['gross'])

  # print (df1_fs['budget'].plot(kind='box'))
  # print (df1_fs['gross'].plot(kind='box'))

  # gross_budget = df1_fs[["gross","budget"]]
  # gross_mean   = mean(x = df1_fs["gross"])
  # budget_mean  = mean(x = df1_fs["budget"])

  # print(gross_budget)
  # print(gross_mean)
  # print(budget_mean)

  # ============================================================ #
  # df1.sample(500)
  # df1.info()
  # df1_fs.info()
  # df1_fs.sample(15)
  # df1.dropna()
  # ============================================================ #
  # sns.heatmap(df1_fs.corr(), annot=Tdrue)
  # sns.pairplot(df1_fs[['budget','gross','genre']])

  # for dirname, _, filenames in os.walk('/kaggle/input'):
  #     for filename in filenames:
  #         print(os.path.join(dirname, filename))
  # warnings.filterwarnings('ignore')
  # %matplotlib inline
  # matplotlib.rcParams['figure.figsize'] = (12,8)
  # pd.options.mode.chained_assignment = None
  # df1 = pd.read_csv('https://drive.google.com/uc?id=1ZpS3-3KqUHV1A5-vdwGDoYaN5RlUwWEc&export=download')
  # df1.head()


  # # print (y)
  # # print(df1.target)
