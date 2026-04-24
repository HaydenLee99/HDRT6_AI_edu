# 쇼핑몰 고객 세분화 연습
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.cluster import KMeans

# 가상의 고객 데이터
np.random.seed(0)
n_customers = 200       # 고객수
annul_spending = np.random.normal(50000, 15000, n_customers)        # 연간 소비액
monthly_visit = np.random.normal(5, 2, n_customers)     # 월 방문 횟수

# 구간 나누기(음수 제거 - clip)
annul_spending = np.clip(annul_spending, 0, None)   # 음수는 0으로 준다
monthly_visit = np.clip(monthly_visit, 0, None)

data = pd.DataFrame({
    'annul_spending':annul_spending,
    'monthly_visit':monthly_visit
})
print(data.head(), data.shape)

# 시각화
plt.scatter(data['annul_spending'], data['monthly_visit'])
plt.xlabel('annul_spending')
plt.ylabel('monthly_visit')
plt.title('소비자 분포')
plt.show()

# K-means 군집화
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(data)

# 군집 결과 시각화
data['clusters'] = clusters

for cluster_id in np.unique(clusters):
    cluster_data = data[data['clusters'] == cluster_id]
    print(data[data['clusters'] == cluster_id].head(3))

    plt.scatter(cluster_data['annul_spending'], cluster_data['monthly_visit'], label=f'군집_{cluster_id}')

# 중심점
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', marker='x', s=200, label='중심점')
plt.xlabel('연간 지출액')
plt.ylabel('한달 방문수')
plt.title('소비자 군집 현황')
plt.legend()
plt.show()