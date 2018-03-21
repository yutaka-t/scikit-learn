# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

# 数値データの読み込み
digits = datasets.load_digits()

# データ形式を確認
print(digits.data)
print(digits.data.shape)

# データの数
n = len(digits.data)

# # 画像と正解値の表示
# images = digits.images
# labels = digits.target
# for i in range(10):
#     # 2行5列 位置は i + 1
#     plt.subplot(2, 5, i + 1)
#     # cmap の指定で白黒表示 ,ピクセル間の補完をnearrest に?
#     plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
#     plt.axis("off")
#     plt.title("Training :" + str(labels[i]))
# plt.show()

# サポートベクターマシーン
clf = svm.SVC(gamma=0.001, C=100.0)
clf.fit(digits.data[:n * 6 // 10], digits.target[:n * 6 // 10])

# # 最後の10個のデータをチェック
# # 正解(マイナスを指定すると末尾からの範囲)
# print(digits.target[-10:])
# # 予測を行う
# print(clf.predict(digits.data[-10:]))

# 残り4割りの画像から、数字を読み取る
# 正解
expected = digits.target[-n * 4 // 10:]
# 予想
predicted = clf.predict(digits.data[-n * 4 // 10:])
# 正解率
print(metrics.classification_report(expected, predicted))
# 誤認識のマトリックス
print(metrics.confusion_matrix(expected, predicted))

# 予測画像の対応(一部)
images = digits.images[-n * 4 // 10:]
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.axis("off")
    plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("Guess: " + str(predicted[i]))
plt.show()
