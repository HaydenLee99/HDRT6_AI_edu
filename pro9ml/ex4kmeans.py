# K-means (K 평균 군집화)
# 비지도 학습 군집화 알고리즘
# 가까운 데이터끼리 K개의 그룹으로 묶어 나누는 방법

# K개 중심점(centroid)을 랜덤으로 정함
# 각 데이터는 가장 가까운 중심에 할당됨
# 각 군집의 평균 위치로 중심을 다시 계산
# 중심이 더 이상 변하지 않을 때까지 반복
# - 중심 이동이 거의 없음
# - 또는 최대 반복 횟수 도달

# 장점
# - 간단하고 빠름
# - 큰 데이터에도 적용 가능

# 단점
# - K를 미리 정해야 함
# - 초기 중심에 따라 결과 달라짐
# - 이상치(outlier)에 민감

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

x, _ = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
print(x[:3])
print(x.shape)

# 시각화
plt.scatter(x[:,0], x[:,1], c='gray', marker='o', s=50)
plt.grid(True)
plt.show()

# K-Means 모델 작성
# cluster의 중심을 선택하는 방법
init_centroid = 'random'            # cluster의 중심을 임의로 선택
init_centroid = 'k-means++'         # cluster의 중심을 k-means++로 선택, 중심을 최대한 멀리함
k_model = KMeans(n_clusters=3, init=init_centroid, random_state=0)
# n_init=10  :  K-means를 10회 실행한다. 가장 좋은 결과를 선택.
pred = k_model.fit_predict(x)
print('pred : ', pred)

# 각 그룹별 보기
# print(x[pred == 0])
# print(x[pred == 1])
# print(x[pred == 2])

print('중심점 : ', k_model.cluster_centers_)

# 시각화
plt.scatter(x[pred == 0, 0], x[pred == 0, 1], c='red', marker='o', s=50, label='cluster_0')
plt.scatter(x[pred == 1, 0], x[pred == 1, 1], c='green', marker='s', s=50, label='cluster_1')
plt.scatter(x[pred == 2, 0], x[pred == 2, 1], c='blue', marker='v', s=50, label='cluster_2')
plt.scatter(k_model.cluster_centers_[:, 0], k_model.cluster_centers_[:, 1], c='black', marker='+', s=60, label='center')
plt.legend()
plt.grid(True)
plt.show()

# K-means의 K값은?      elbow or silhoutte 기법을 이용해 K값 얻기
def elbow(x):
    sse = []
    for k in range(1,11):
        km = KMeans(n_clusters=k, init=init_centroid, random_state=0)
        km.fit(x)
        sse.append(km.inertia_)
    plt.plot(range(1,11), sse, marker='o')
    plt.xlabel('군집수')
    plt.ylabel('SSE')
    plt.show()

elbow(x)
# k는 3을 주는게 좋겠다 판단 가능

# 실루엣(silhouette) 기법
# 클러스터링의 품질을 정량적으로 계산해 주는 방법이다.
# 클러스터의 개수가 최적화되어 있으면 실루엣 계수의 값은 1에 가까운 값이 된다.
# 실루엣 기법은 k-means 클러스터링 기법 이외에 다른 클러스터링에도 적용이 가능하다

from sklearn.metrics import silhouette_samples
from matplotlib import cm

# 데이터 X와 X를 임의의 클러스터 개수로 계산한 k-means 결과인 y_km을 인자로 받아 각 클러스터에 속하는 데이터의 실루엣 계수값을 수평 막대 그래프로 그려주는 함수를 작성함.
# y_km의 고유값을 멤버로 하는 numpy 배열을 cluster_labels에 저장. y_km의 고유값 개수는 클러스터의 개수와 동일함.

def plotSilhouette(x, pred):
    cluster_labels = np.unique(pred)
    n_clusters = cluster_labels.shape[0]   # 클러스터 개수를 n_clusters에 저장
    sil_val = silhouette_samples(x, pred, metric='euclidean')  # 실루엣 계수를 계산
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        # 각 클러스터에 속하는 데이터들에 대한 실루엣 값을 수평 막대 그래프로 그려주기
        c_sil_value = sil_val[pred == c]
        c_sil_value.sort()
        y_ax_upper += len(c_sil_value)

        plt.barh(range(y_ax_lower, y_ax_upper), c_sil_value, height=1.0, edgecolor='none')
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_sil_value)

    sil_avg = np.mean(sil_val)         # 평균 저장

    plt.axvline(sil_avg, color='red', linestyle='--')  # 계산된 실루엣 계수의 평균값을 빨간 점선으로 표시
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('클러스터')
    plt.xlabel('실루엣 개수')
    plt.show() 

# 그래프를 보면 클러스터 1~3 에 속하는 데이터들의 실루엣 계수가 0으로 된 값이 아무것도 없으며, 
# 실루엣 계수의 평균이 0.7 보다 크므로 잘 분류된 결과라 볼 수 있다.

X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
km = KMeans(n_clusters=3, random_state=0) 
y_km = km.fit_predict(X)

plotSilhouette(X, y_km)