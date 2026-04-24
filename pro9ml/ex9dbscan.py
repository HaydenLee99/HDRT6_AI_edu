# DBSCAN(Density-Based Spatial Clustering of Applications with Noise)
# : 밀도 기반 군집화 알고리즘

# - 데이터가 조밀하게 모여 있는 영역을 하나의 군집으로 판단한다.
# - 밀도가 낮은 점들은 군집에 속하지 않는 노이즈(noise)로 처리한다.
# - K-means와 달리 군집 개수(k)를 미리 정할 필요가 없다.

# 주요 파라미터
# - eps : 한 점 주변에서 이웃으로 인정할 반경
# - min_samples : 해당 반경 안에 있어야 하는 최소 데이터 수

# 점의 종류
# - core point   : 주변에 이웃이 충분히 많은 중심 점
# - border point : core point 근처에 있지만 스스로는 중심이 아닌 점
# - noise point  : 어떤 군집에도 속하지 않는 점

# 장점
# - 군집 개수를 미리 정하지 않아도 된다.
# - 이상치 탐지에 유리하다.
# - 원형이 아닌 복잡한 모양의 군집도 찾을 수 있다.

# 단점
# - eps, min_samples 값 설정에 민감하다.
# - 군집마다 밀도 차이가 크면 성능이 떨어질 수 있다.
# - 고차원 데이터에서는 거리 기반 성능이 약해질 수 있다.

# 결과 해석
# - 같은 번호 label : 같은 군집
# - label = -1 : 노이즈 데이터

import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib import style
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

# 샘플 데이터
x, y = make_moons(n_samples=200, noise=0.05, random_state=0, shuffle=True)
print(x[:5], x.shape)
# print(y)

# 시각화
plt.scatter(x[:,0], x[:,1], c=y)
plt.show()

# KMeans 군집 분류 진행
km = KMeans(n_clusters=2, random_state=0, init='k-means++')
pred1 = km.fit_predict(x)
print('km 예측 군집 id : ', pred1[:10])

# km 결과 시각화
def plotResult(x, pr, bymsg):
    plt.scatter(x[pr==0, 0], x[pr==0, 1], c='blue', marker='o', s=40, label='cluster 1')
    plt.scatter(x[pr == 1, 0], x[pr == 1, 1], c='red', marker='s', s=40, label='cluster 2')
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='black', marker='+', s=40, label='centroid')
    plt.title(f'Clustering Result by {bymsg}')
    plt.legend()
    plt.show()

plotResult(x, pred1, 'K-means')
# KMeans 군집 분류 결과 : 분류 불가능

# DBSCAN 방식으로 군집 분류 진행
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
# eps : 샘플간 최대 거리 설정값
# min_samples : 데이터에 대해 이웃한(반경 포인트들의) 최소 샘플수
pred2 = db.fit_predict(x)
print('DBSCAN 예측 군집 id : ', pred2[:10])
print('DBSCAN 군집 종류 : ', set(pred2))       # 이상치 확인용 : 확인 결과 이상치 없음
plotResult(x, pred2, 'DBSCAN')

