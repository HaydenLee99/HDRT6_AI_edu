# sklearn이 제공하는 단층 신경망 Perceptron
# 이항 분류 가능
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
label_and = np.array([0,0,0,1])
label_or = np.array([0,1,1,1])
label_xor = np.array([0,1,1,0])
label_imp = np.array([1,1,0,1])

max_iter = 1000
print('max_iter 설정값 : ', max_iter)
ml_and = Perceptron(max_iter=max_iter).fit(feature, label_and)
ml_or = Perceptron(max_iter=max_iter).fit(feature, label_or)
ml_xor = Perceptron(max_iter=max_iter).fit(feature, label_xor)
ml_imp = Perceptron(max_iter=max_iter).fit(feature, label_imp)

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
# Perceptron 설명:
# Perceptron은 딥러닝에서 사용하는 경사하강법과 달리 "틀린 것만 고치는" 알고리즘을 사용.
# 예측값이 틀리면 가중치를 갱신하고, 맞으면 통과. 이를 max_iter만큼 반복해서 학습.
# 이 방식은 선형 회귀식이나 로지스틱 회귀에서 사용하는 방식과 유사.