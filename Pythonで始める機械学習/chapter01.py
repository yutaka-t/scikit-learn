# -*- coding : utf-8 -*-
# ====================================================================
# 本の前提条件
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy as sp
import sklearn

# from IPython.display import display
# plot.show で画面を出す

print("Python version : {}".format(sys.version))
print("Pandas version : {}".format(pd.__version__))
print("matplotlib version : {}".format(matplotlib.__version__))
print("numpy version : {}".format(np.__version__))
print("scipy version : {}".format(sp.__version__))
# print("IPython version : {}".format(IPython.__version__))
print("sklearn version : {}".format(sklearn.__version__))

# ====================================================================
print("{}".format("#"*20))
print("\nChapter 01\n")

# P.18 まとめの内容
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# データ読み込み
iris_dataset = load_iris()

# データを分割
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# 学習
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# 結果確認
# テスト用の値（X_test) を与えて、答え(y_test)との一致を確認している
print("Test set score : {:.2f}".format(knn.score(X_test, y_test)))
