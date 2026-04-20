# KNN (K-Nearest Neighbors, 최근접 이웃 알고리즘)
# 새로운 데이터가 들어왔을 때, 기존 데이터 중 가장 가까운 K개의 데이터를 찾아 그들의 "다수결(또는 평균)"로 결과를 결정하는 알고리즘

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

train = [
    [5, 3, 2],
    [1, 3, 5],
    [4, 5, 6]
]
label = [0, 1, 1]

plt.plot(train, 'o')
plt.xlim([-1, 5])
plt.ylim([0, 8])
plt.show()

kmodel = KNeighborsClassifier(n_neighbors=3, weights='distance')
kmodel.fit(train, label)
pred = kmodel.predict(train)
print('pred : ', pred)
print(f'test acc : {kmodel.score(train, label)}')

new_data = [[1, 2, 9], [6, 2, 1]]
new_pred = kmodel.predict(new_data)
print('new_pred : ', new_pred)