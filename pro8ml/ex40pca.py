# PCA(주성분 분석)는 입력 데이터의 공분산 행렬을 고유값 분해하여
# 얻어진 고유벡터(주성분 방향)에 데이터를 투영하여 차원을 축소하는 방법.

# 이때 고유벡터는 데이터의 분산이 가장 큰 방향을 의미하며,
# 중요도가 높은 방향부터 차례대로 선택하여 고차원 데이터를 저차원으로 변환함.

# 데이터의 정보를 최대한 유지하면서(분산 최대 보존)
# 차원을 축소하는 것이 목적이다.

# iris data로 차원 축소 해보기
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
n=10
x=iris.data[:n,:2]

# 시각화 1 : 각 sample의 두 특성값을 선으로 연결하여 비교
# plt.plot(x.T, 'o:')
# plt.xticks([0,1], ['꽃받침길이', '꽃받침너비'])
# plt.grid(True)
# plt.title('아이리스 크기 특성')
# plt.xlabel('특성의 종류')
# plt.ylabel('특성값')
# plt.xlim(-0.5,2)
# plt.ylim(2.5,6)
# plt.legend(['표본 {}'.format(i+1) for i in range(n)])
# plt.show()

# 시각화 2 : 산점도
df = pd.DataFrame(x)
print(df)

ax = sns.scatterplot(x=df[0], y=df[1], marker="s", s=100, color="b")
for i in range(n):
    ax.text(x[i,0]-0.05, x[i,1]+0.03,'표본{}'.format(i+1))
plt.xlabel("꽃받침 길이")
plt.ylabel("꽃받침 폭")
plt.title("iris 특성")
plt.axis("equal")
# plt.show()

# 위 시각화 결과 두 변수는 공통적인 특징이 있으므로 차원축소의 근거가 있다고 판단.
# PCA를 진행 : 선형변환을 통해 차원을 축소  from sklearn.decomposition import PCA

# step 0 : 데이터 평균 제거
# 각 변수의 평균을 0으로 만들어야 공분산이 의미를 가짐 (centered data)

# step 1 : 입력데이터의 공분산 행렬을 생성
# Cov(X) = (1/n) * X^T X
# X^T X = 내적행렬 -> 만일 I라면 X는 직교행렬

# step 2 : 공분산 행렬의 고유벡터와 고유값 계산
# 고유벡터 = 데이터가 가장 많이 퍼진 방향 (주성분 방향)
# 고유값 = 해당 방향으로의 분산 크기

# step 3 : 고유값이 큰 순으로 k개(PCA 변환 차수)만큼 고유벡터 선택
# 분산이 큰 방향일수록 정보가 많다고 판단

# step 4 : 선택된 고유벡터를 이용해 입력 데이터를 변환
# Z = X · W (W는 선택된 고유벡터 행렬)
# → 고차원 데이터를 저차원으로 투영

pca1 = PCA(n_components=1)
# n_components=1 변환할 차원 수 입력
# pca1.components_          : 고유벡터 (주성분 방향)
# pca1.explained_variance_  : 고유값  (설명된 분산)
# pca1.explained_variance_ratio_  : 비율 (PCA 후 정보(분산) 보존 비율)

x_low = pca1.fit_transform(x)
# 특징 행렬을 낮은 차원의 근사행렬로 변환
print('x_low : ', x_low, ' ', x_low.shape)
# 주성분 값 원복하기
x2 = pca1.inverse_transform(x_low)
print('원복 후 x 값 : ', x2, ' ',x2.shape)
# 원복해도 일부의 값을 잃기에 완전히 같은 값으로 돌아가지 못함

# 주성분 분석값 시각화
pc1 = pca1.components_[0]
mean = x.mean(axis=0)       # 데이터 평균 (중심점)

df=pd.DataFrame(x2)
ax = sns.scatterplot(x=df[0], y=df[1], marker="s", s=100, color="b")
for i in range(n):
    ax.text(x[i,0]-0.05, x[i,1]+0.03,'표본{}'.format(i+1))
#pca 축 화살표
plt.quiver(
    mean[0], mean[1],    # 시작점 평균
    pc1[0], pc1[1],      # 방향
    scale=3, color='r', width=0.01
    )
plt.xlabel("꽃받침 길이")
plt.ylabel("꽃받침 폭")
plt.title("iris 특성")
plt.axis("equal")
plt.grid(True)
# plt.show()

# iris data 4개의 열을 차원축소해 2개의 열로 변환 후 SVM 분류모델 만들기
x=iris.data
print('x[0, :] : ',x[0, :],' ', x[0, :].shape)
pca2 = PCA(n_components=2)
x_low2 = pca2.fit_transform(x)
print('x_low2 : ', x_low2[0, :],' ', x_low2.shape)

# 변동성 비율 확인
print(pca2.explained_variance_ratio_)

# 차원 복귀
x4 = pca2.inverse_transform(x_low2)
print('x4 : ', x4[0, :],' ', x4.shape)

iris1 = pd.DataFrame(x, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
print(iris1.head(3))

iris2 = pd.DataFrame(x_low2, columns=['var1', 'var2'])
print(iris2.head(3))

from sklearn import svm, metrics
feature1 = iris1.values
label = iris.target

print('원본 데이터 이용한 SVM 분류 모델')
model1 = svm.SVC(C=0.1, random_state=0).fit(feature1, label)
pred1 = model1.predict(feature1)
print('model1 accuracy : ', metrics.accuracy_score(label, pred1))

feature2 = iris2.values
label = iris.target
print('pca 이용한 SVM 분류 모델')
model2 = svm.SVC(C=0.1, random_state=0).fit(feature2, label)
pred2 = model2.predict(feature2)
print('model2 accuracy : ', metrics.accuracy_score(label, pred2))

