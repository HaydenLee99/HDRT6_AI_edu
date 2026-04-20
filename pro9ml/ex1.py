# 비지도 학습 - 군집 분석
# 군집 분석(Clustering)은 주어진 데이터들을 그룹화하는 기법으로, 
# 데이터를 유사성에 따라 그룹으로 나눈다.
# 알고리즘은 K-means, 계층적 군집화(Hierarchical Clustering)...

# 군집 분석의 핵심은 '데이터 간의 유사도' 정의.
# 유사도는 보통 거리 측정 방식(유클리드 거리, 맨해튼 거리 등)이나 상관계수를 통해 계산.

# 거리나 상관계수를 사용하여 비슷한 특성을 가진 객체들을 그룹으로 만들고, 
# 그룹 간에 어떤 차이가 있는지 분석.

# 예를 들어, 군집 분석을 통해 데이터를 군집화한 후, 
# 그룹 간 평균 차이를 분석하려면 t-test나 ANOVA 분석 사용.

# 군집 분석은 고객 세분화, 이상치 탐지, 이미지 분류 등의 분야에서 활용.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

np.random.seed(123)
var=['x','y']
labels=['점0','점1','점2','점3','점4']
x=np.random.random_sample([5,2]) * 10
df = pd.DataFrame(x,columns=var, index=labels)
print(df)

plt.scatter(x[:, 0], x[:, 1], c='b', marker='o', s=50)
for i, txt in enumerate(labels):
    plt.text(x[i,0], x[i,1], txt)
plt.grid(True)
plt.show()

# 각 점 간의 거리 계산
from scipy.spatial.distance import pdist, squareform
dist_vec = pdist(df, metric='euclidean')
print(dist_vec)
# pdist의 결과를 squareform으로 보기
row_dist = pd.DataFrame(squareform(dist_vec), columns=labels, index=labels)
print(row_dist)

from scipy.cluster.hierarchy import linkage
# linkage : 응집형 계층적 군집분석
row_clusters = linkage(dist_vec, method='ward')
df2 = pd.DataFrame(row_clusters, columns=['클러스터id1','클러스터id2','거리', '클러스터 멤버'])

# 클러스터의 계층구조를 계통도로 출력
from scipy.cluster.hierarchy import dendrogram
dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.show()
