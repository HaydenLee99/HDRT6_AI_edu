# 결정 트리(Decision Tree) 분류 모델 개념

# 1. 정의
# - 결정 트리는 데이터를 분류하거나 예측하기 위해 '트리 구조'를 사용하는 모델
# - 각 노드는 '특징(feature)에 대한 조건'으로 데이터를 분할
# - 잎 노드(Leaf Node)에는 최종 클래스 레이블이나 예측값이 배정됨

# 2. 분할 기준
# - 데이터 균일도(homogeneity)를 최대화하는 방향으로 분할
# - 즉, 한 노드 안의 데이터가 한 클래스에 최대한 몰리도록 나눔
# - 대표적인 불순도(impurity) 지표:
#   1) 지니 불순도(Gini Impurity)
#      - 노드 내 클래스 혼합 정도를 0~1로 계산
#      - 낮을수록 한 클래스가 많이 포함됨
#   2) 엔트로피(Entropy)
#      - 정보 이득(Information Gain)을 최대화하는 분할 선택
#      - 불확실성 감소량 기준

# 3. 트리 생성 과정
# 1) 루트 노드(Root Node) : 전체 데이터를 기준으로 시작
# 2) 분할(Split) : 각 특징과 임계값을 시험하여 불순도 최소화
# 3) 자식 노드 생성 : 조건에 맞게 데이터를 분할
# 4) 반복 : 각 자식 노드에서도 동일하게 분할
# 5) 종료 조건:
#    - 최대 트리 깊이(max_depth)에 도달
#    - 노드의 데이터 수(min_samples_split) 미만
#    - 불순도가 더 이상 감소하지 않을 때

# 4. 예측 과정
# - 새로운 데이터가 루트 노드에서 시작
# - 노드 조건에 따라 좌우 자식으로 이동
# - 최종 잎 노드에 도달하면 그 노드의 클래스가 예측 결과

# 5. 특징
# - 직관적이고 이해하기 쉬움
# - 특징 중요도(feature importance)를 확인 가능
# - 과적합(overfitting)에 취약 → 트리 깊이 제한, 최소 샘플 수 조정 등 필요

# 6. 요약
# - 결정 트리는 '데이터 균일도를 최대화하는 규칙'으로 데이터를 분할하는 모델
# - 각 질문(노드 조건)이 데이터를 절반씩 나누어 정보 이득을 최대화하는 전략과 유사
# - 아키네이터나 20Q 게임처럼, 반복적 질문으로 후보군을 줄이는 원리와 닮음

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np

# make_classification으로 분류용 데이터 생성
x, y = make_classification(
    n_samples=100,        # 생성할 데이터 샘플 개수
    n_features=2,         # 각 데이터 포인트의 총 특징(feature) 수
    n_redundant=0,        # 다른 특징으로 선형 조합된 불필요한 특징 수: 0개 (모든 특징 독립적)
    n_informative=2,      # 실제 분류에 영향을 주는 중요한 특징 수: 2개 (모두 유용)
    random_state=42       # 랜덤 시드 고정
)

model = DecisionTreeClassifier(criterion='gini', max_depth=3)
model.fit(x,y)      # 지도학습이기에 feature와 label을 준다.

# 트리 구조 시각화
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=['x1','x2'], class_names=['0','1'], filled=True)
plt.show()

# 결정경계 시각화
xx, yy = np.meshgrid(   # x축, y축 값을 조합해소 좌표 격자를 생성
    # x1 범위를 100개의 구간으로 나눔
    np.linspace(x[:,0].min(), x[:,1].max(),100),
    # x2 범위를 100개의 구간으로 나눔
    np.linspace(x[:,1].min(), x[:,0].max(),100)
)

# 모든 좌표에 대한 예측값 계산
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)     # 예측 결과를 원래 grid로 변환
print(z)

plt.contour(xx, yy, z, alpha=0.3)   # 영역을 색으로 채워 결정경계 표현
plt.scatter(x[:,0], x[:,1],c=y)
plt.title("Decision boundary")
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()