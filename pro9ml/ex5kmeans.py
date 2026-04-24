# 비계층적 군집 분석 : K-means
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.cluster import KMeans

student = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
scores = np.array([76,95,65,85,60,69,55,88,83,72]).reshape(-1,1)

# K=3
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)
km_clusters = kmeans.fit_predict(scores)
print(km_clusters)

df = pd.DataFrame({
    'student':student,
    'score':scores.ravel(),
    'cluster':km_clusters
})
print(df)

# 군집별 평균 점수
grouped = df.groupby('cluster')['score'].mean()
print(grouped)

# 시각화
x_position = np.arange(len(student))
y_scores = scores.ravel()
colors = {0:'red', 1:'blue', 2:'green'}

plt.figure(figsize=(10,6))
# 학생별 군집 색깔로 구분한 산점도 출력
for i, (x,y,cluster) in enumerate(zip(x_position, y_scores, km_clusters)):
    plt.scatter(x,y,c=colors[cluster],s=100)
    plt.text(x,y+1.5, student[i], fontsize=10,ha='center')

# 중심점
centers = kmeans.cluster_centers_
for center in centers:
    plt.scatter(len(student)//2, center[0], marker='X', c='black', s=200)

plt.xticks(x_position, student)
plt.xlabel('학생명')
plt.ylabel('점수')
plt.grid(True)
plt.show()