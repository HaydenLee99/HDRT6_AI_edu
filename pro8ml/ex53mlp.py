# MLP (Multi-Layer Perceptron) 핵심 정리
# 이미지 처리, 자연어 처리, 예측 모델링 등에 주로 사용

# 다층 신경망(Multi-Layer Neural Network)으로 입력층, 은닉층, 출력층으로 이루어짐

# 주요 활성화 함수:
#  ReLU: 음수는 0으로, 양수는 그대로 출력
#  Sigmoid: 출력값을 0과 1 사이로 제한
#  tanh: 출력값을 -1과 1 사이로 제한
#  Softmax: 출력층에서 각 클래스의 확률을 계산

# 순방향 전파(Feedforward): 입력 데이터가 각 층을 거쳐 순차적으로 전달되며, 최종 출력값을 계산
# 역전파(Backpropagation): 예측값과 실제값 사이의 오차를 계산하고, 이 오차를 역방향으로 전파하여 가중치(weight)를 업데이트하는 방식으로 
#                         모델 학습. 역전파로 계산된 오차를 기반으로 가중치를 경사하강법을 이용해 최적화.

# MLP 장점:
#  다층 신경망 구조로 비선형 문제 해결 가능

# MLP 한계:
#  모델이 너무 복잡해져 훈련 데이터에 과적합될 수 있음
#  훈련 시간이 길어질 수 있음

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
label_and = np.array([0,0,0,1])
label_or = np.array([0,1,1,1])
label_xor = np.array([0,1,1,0])
label_imp = np.array([1,1,0,1])

max_iter = 500      # 추천횟수는 500 ~ 1000정도
print('max_iter 설정값 : ', max_iter)
ml = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=10, solver='adam', learning_rate_init=0.01, verbose=1)
ml_and = ml.fit(feature, label_and)
ml_or = ml.fit(feature, label_or)
ml_xor = ml.fit(feature, label_xor)
ml_imp = ml.fit(feature, label_imp)

pred_and = ml_and.predict(feature)
print('And 연산')
print('실제값 : ', label_and)
print('예측값 : ', pred_and)
print('acc_and : ', accuracy_score(label_and, pred_and))

pred_or = ml_or.predict(feature)
print('Or 연산')
print('실제값 : ', label_or)
print('예측값 : ', pred_or)
print('acc_or : ', accuracy_score(label_or, pred_or))

pred_xor = ml_xor.predict(feature)
print('Xor 연산')
print('실제값 : ', label_xor)
print('예측값 : ', pred_xor)
print('acc_xor : ', accuracy_score(label_xor, pred_xor))

pred_imp = ml_imp.predict(feature)
print('Implication 연산')
print('실제값 : ', label_imp)
print('예측값 : ', pred_imp)
print('acc_imp : ', accuracy_score(label_imp, pred_imp))

# 일반 자료 분류
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

x, y = make_moons(n_samples=300, noise=0.2, random_state=42)
print(x[:5], y[:5])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = MLPClassifier(
    hidden_layer_sizes=(10,10),
    solver='adam',
    max_iter=1000,
    random_state=42,
    activation='relu')
model.fit(x_train, y_train)
pred = model.predict(x_test)
print('acc : ', accuracy_score(y_test, pred))