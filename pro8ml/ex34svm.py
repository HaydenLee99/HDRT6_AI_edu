# SVM은 데이터를 나누는 "최적의 경계선"을 찾는 거리 기반 모델.
# 목표: 두 클래스 사이를 가장 넓게 떨어뜨려서 나누는 것 (margin 최대화)
# 특징: 거리 기반 모델이므로 스케일링 필수
#      작은, 중간 데이터에서 강력한 성능        데이터가 많으면 느린 속도

# margin: 경계선과 데이터 사이의 여유 공간으로 이 공간이 클수록 일반화 성능이 좋아짐
# support vector: 경계선을 실제로 결정하는 가장 중요한 데이터들로 이 데이터들만 모델에 영향을 줌
# kernel: 직선으로 나눌 수 없는 데이터를 고차원으로 변환해서 나눌 수 있게 만드는 방법

# C: 오차를 얼마나 허용할지 결정                    gamma: 한 데이터가 영향을 미치는 범위    
#   작으면 부드러운 경계 (일반화)                          작으면 넓게 영향 (단순한 경계)           
#   크면 정확하게 맞추려 함 (과적합 위험)                   크면 좁게 영향 (복잡한 경계)    

# Support vector 확인해보기 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='malgun gothic')

X, y = make_blobs(n_samples=50, centers=2, cluster_std=0.5, random_state=4)
y = 2 * y - 1

plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', label="-1 클래스")
plt.scatter(X[y == +1, 0], X[y == +1, 1], marker='x', label="+1 클래스")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("학습용 데이터")
plt.show()

from sklearn.svm import SVC
model = SVC(kernel='linear', C=1.0).fit(X, y)  # tuning parameter  값을 변경해보자.

xmin = X[:, 0].min()
xmax = X[:, 0].max()

ymin = X[:, 1].min()
ymax = X[:, 1].max()

xx = np.linspace(xmin, xmax, 10)
yy = np.linspace(ymin, ymax, 10)
X1, X2 = np.meshgrid(xx, yy)
z = np.empty(X1.shape)

for (i, j), val in np.ndenumerate(X1):    # 배열 좌표와 값 쌍을 생성하는 반복기를 반환
    x1 = val
    x2 = X2[i, j]
    p = model.decision_function([[x1, x2]])
    z[i, j] = p[0]

plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', label="-1 클래스")
plt.scatter(X[y == +1, 0], X[y == +1, 1], marker='x', label="+1 클래스")
plt.contour(X1, X2, z, levels=[-1, 0, 1], colors='k', linestyles=['dashed', 'solid', 'dashed'])
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, alpha=0.3)

x_new = [10, 2]
plt.scatter(x_new[0], x_new[1], marker='^', s=100)
plt.text(x_new[0] + 0.03, x_new[1] + 0.08, "테스트 데이터")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("SVM 예측 결과")
plt.show()

# Support Vectors 값 출력
print(model.support_vectors_)

# 참고 : scikit-learn에서 많이 사용하는 인터페이스 중 하나는 분류기에 예측의 불확실성을 추정할 수 있는 기능이다. 
# 어떤 테스트 포인트에 대해 분류기가 예측한 클래스가 무엇인지 뿐만 아니라 정확한 클래스임을 얼마나 확신하는지가 중요할 때가 많다. 
# 예를 들어 보안, 의료 등의 분야가 그렇다. (틀린 예측의 결과가 심각한 피해를 만들 가능성이 있는 경우)
# scikit-learn 분류기에서 불확실성을 추정할 수 있는 함수가 두 개 있다. decision_function과 predict_proba이다. 
# 대부분의 분류 클래스는 하나 이상을 제공한다.