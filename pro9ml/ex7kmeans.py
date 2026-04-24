# 비계층적 군집 분석 - K-means - iris dataset으로 최종 정리
# 군집 분석, 정량 평가, 군집별 평균 비교(ANOVA)
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
# adjusted_rand_score : 군집 vs 실제 라벨 비교
# normalized_mutual_info_score : 정보량 기반 유사도
# silhouette_score : 군집 자체 품질 평가
from sklearn.decomposition import PCA       # 4차원을 2차원으로 압축하여 시각화

iris = load_iris()
x = iris.data
y = iris.target
feature_names = iris.feature_names

df = pd.DataFrame(x, columns=feature_names)
# print(df.head(3))

# scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
# print(x_scaled[:2])

# K-means 모델
k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
clusters = kmeans.fit_predict(x_scaled)

df['cluster'] = clusters
print('클러스터 중심 값 : ', kmeans.cluster_centers_)

# PCA : 시각화용
pca = PCA(n_components=2)       # 2차원으로 축소
x_pca = pca.fit_transform(x_scaled)
print('변수 보존 평가 : ', pca.explained_variance_ratio_.sum())
# 원본 데이터의 분산(정보)을 얼마나 보존했는지 나타내는 비율 : 약 95.8%

# PCA 기반 시각화 : 4개 열을 2차원 차트에 표현하기 위함
plt.figure(figsize=(6,5))
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=clusters, palette='Set1')
plt.title('K-means Clustering')
plt.xlabel('주성분_1')
plt.ylabel('주성분_2')
plt.show()

# 실제 label과 군집 비교: 교차표
ct = pd.crosstab(y, clusters)
print(ct)

for i in range(ct.shape[0]):
    max_cluster = ct.iloc[i].idxmax()
    print(f'실제 클래스 {i} -> 군집 {max_cluster}')

# 정량 평가
ari = adjusted_rand_score(y, clusters)
nmi = normalized_mutual_info_score(y, clusters)
sil_score = silhouette_score(x_scaled, clusters)

print('ARI : ', ari)
print('NMI : ', nmi)
print('silhouette_score : ', sil_score)


inertia_list=[]
k_range = range(2,11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(x_scaled)
    inertia_list.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(k_range, inertia_list, marker='o')
plt.title('엘보우 기법')
plt.xlabel('클러스터 수(k)')
plt.ylabel('inertia')
plt.show()

# 실제와 군집 비교 시각화
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=y, palette='Set1')
plt.title('실제 라벨')

plt.subplot(1,2,2)
sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=clusters, palette='Set1')
plt.title('군집 결과')
plt.show()

# 클러스터별 평균 분석
cluster_mean = df.groupby('cluster').mean()
print('클러스터별 평균 : ', cluster_mean)

# 군집 3개 : 군집간 평균차이 검정(ANOVA)
# 귀무 : 군집간 평균의 차이가 없다.
# 대립 : 군집간 평균의 차이가 있다.
from scipy.stats import f_oneway
for col in feature_names:
    group0 = df[df['cluster'] == 0][col]
    group1 = df[df['cluster'] == 1][col]
    group2 = df[df['cluster'] == 2][col]
    # ANOVA
    f_stat, p_val = f_oneway(group0, group1, group2)
    print(f'{col} : f-statistic:{f_stat:.4f}, p-value:{p_val:.4f}')

    # 해석
    if p_val >= 0.05:
        print("유의미한 평균 차이를 확인할 수 없다.")
    else:
        print("적어도 한 군집은 평균 차이가 존재한다.")

# K-means가 꽃받침, 꽃잎 길이/너비를 어느정도 반영한 군집 분석했음을 알 수 있음.

# 사후검정 - petal length로 작업
from statsmodels.stats.multicomp import pairwise_tukeyhsd
feature = 'petal length (cm)'
tukey = pairwise_tukeyhsd(endog=df[feature], groups=df['cluster'], alpha=0.05)
print('petal length (cm) tukeyhsd 결과 :\n', tukey)
# petal length (cm) tukeyhsd 결과 :
# Multiple Comparison of Means - Tukey HSD, FWER=0.05
# ===================================================
# group1 group2 meandiff p-adj  lower   upper  reject
# ---------------------------------------------------
#      0      1  -2.9078   0.0 -3.1405 -2.6751   True
#      0      2   1.1408   0.0  0.9043  1.3773   True
#      1      2   4.0486   0.0  3.8088  4.2884   True
# ---------------------------------------------------

# 사후검정 시각화
tukey.plot_simultaneous(figsize=(6,4))
plt.title(f'tukeyhsd - {feature}')
plt.xlabel('평균 차이')
plt.show()
# 모든 군집 쌍에서 평균 차이가 유의함 (petal length 기준 군집 분리 잘 됨)

# 군집별 박스플랏
for col in feature_names:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='cluster', y=col, data=df)
    plt.title(f'{col} by cluster')
    plt.show()
cluster_mean['label'] = ['Type A', 'Type B', 'Type C']
