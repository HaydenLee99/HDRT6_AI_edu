# 계측정 군집분석
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
student = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10']
scores = np.array([76,95,65,85,60,69,55,88,83,72]).reshape(-1,1)

linked = linkage(scores, method='ward')

plt.figure(figsize=(10,6))
dendrogram(linked,labels=student)
plt.title('student score')
plt.xlabel('student')
plt.ylabel('distance')
plt.axhline(y=25, color='red',linestyle='--')
plt.legend()
plt.grid()
plt.show()

# 군집 3개로 나누기
clusters = fcluster(linked, t=3, criterion='maxclust')
print(clusters)

for stu, cluster in zip(student, clusters):
    print(f'{stu} : cluster {cluster}')

# 군집별로 점수와 이름 확인
cluster_info = {}
for student, cluster, score in zip(student, clusters, scores.flatten()):
    if cluster not in cluster_info:
        cluster_info[cluster]={'student':[], 'scores':[]}
    cluster_info[cluster]['student'].append(student)
    cluster_info[cluster]['scores'].append(score)
print(cluster_info)

# 군집별로 평균점수와 이름 확인
for cluster_id, info in sorted(cluster_info.items()):
    avg_score = np.mean(info['scores'])
    student_list = ', '.join(info['student'])
    print(f'Cluster {cluster_id}:평균점수={avg_score:.2f}, 학생들={student_list}')

# 군집별 산점도
x_positions = np.arange(len(student))
y_scores=scores.ravel()
colors = {1:'red',2:'blue',3:'green'}
plt.figure(figsize=(10,6))
for i, (x,y,cluster) in enumerate(zip(x_positions, y_scores, clusters)):
    plt.scatter(x,y,color=colors[cluster], s=100)
    plt.text(x,y+1.5, student[i], fontsize=12, ha='center')
plt.xticks(x_positions, student)
plt.xlabel('student')
plt.ylabel('score')
plt.title('score cluster')
plt.grid(True)
plt.show()