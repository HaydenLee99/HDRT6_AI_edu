# 계층적 군집 분석 - iris dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# adjusted_rand_score : ARI(Adjusted Random Index) - 두 점이 같은 군집인지 아닌지 쌍으로 비교
# normalized_mutual_info_score : NMI(Normalized Mutual Information) - 정보 이론 기준으로 얼마나 비슷한가
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

iris = load_iris()
x = iris.data
y = iris.target
labels = iris.target_names

df = pd.DataFrame(x, columns=iris.feature_names)
print(df.head(3))

# scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 계층적 군집
z = linkage(x_scaled, method='ward')

# dendrogram 확인
plt.figure(figsize=(16,5))
dendrogram(z)
plt.title('iris로 계층적 군집')
plt.xlabel('sample data')
plt.ylabel('유클리드 거리')
plt.show()

# dendrogram을 잘라서 최대 3개의 군집 만들기
clusters = fcluster(Z=z, t=3, criterion='maxclust')
df['cluster']=clusters
print(df.head())
print(df.tail())

# 2개 feature 시각화
plt.figure(figsize=(6,5))
sns.scatterplot(x=x_scaled[:,0], y=x_scaled[:,1], hue=clusters, palette='Set1')
# hue=clusters 군집 결과에 따라 색을 다르게 표시
# palette='Set1' 색상 스타일 지정
plt.title('계층적 군집 결과')
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.show()      # 꽤 비슷하게 나눠짐

print('실제 label : ', y[:10])
print('군집 결과 : ', clusters[:10])
# 실제 0이 군집 1로 군집화 되었음을 확인 가능

print('군집 결과 검증용 교차표:\n')
ct = pd.crosstab(y, clusters)
print(ct)
# col_0   1   2   3     군집 label
# row_0                 실제 label
# 0      49   1   0
# 1       0  27  23
# 2       0   2  48
# setosa(1)는 완벽히 분리
# versicolor(2)와 virginica는 일부 섞인 결과를 보임

print('교차표 보조 설명 : 각 실제 class가 가장 많이 속한 군집:')
for i in range(ct.shape[0]):
    max_cluster = ct.iloc[i].idxmax()
    print(f'실제 class {i} -> 군집 {max_cluster} (갯수:{ct.iloc[i].max()})')

# 정량적 평가 : 군집 결과가 실제 정답과 얼마나 유사한지를 수치로 표현
ari = adjusted_rand_score(y, clusters)
print('평가지표(ARI):', ari)
# 해석 기준 : 
# ARI 값      의미               
# 1.0        완전히 동일한 군집       
# 0.8 ~ 1.0  매우 잘된 군집         
# 0.5 ~ 0.8  꽤 잘된 군집          
# 0.2 ~ 0.5  보통 수준            
# 0 ~ 0.2    약한 유사성           
# 0          랜덤 수준            
# < 0        랜덤보다 못함 (잘못된 군집) 


nmi = normalized_mutual_info_score(y, clusters)
print('평가지표(NMI):', nmi)
# 해석 기준 : 
# NMI 값        의미   
# 1.0      완전 일치
# 0.6~1.0  강한 유사
# 0.3~0.6  보통   
# 0~0.3    약함   
# 0        랜덤   
