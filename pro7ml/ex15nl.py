# 비선형회귀분석(Non-Linear Regression)
# 직선의 회귀선을 곡선으로 변환해 보다 정확하게 데이터 변화를 예측하는데 목적이 있음
# 비정규성 상황에서 대체할 수 있는 방법으로 다항식 항을 추가한 다항회귀 모델을 사용함

import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.metrics import r2_score

x=np.array([1,2,3,4,5])
y=np.array([4,2,1,3,7])

# 선형회귀 모델을 적용한다면?
from sklearn.linear_model import LinearRegression
x = x[:, np.newaxis]        # 차원 확대  (1차원 -> 2차원)
model = LinearRegression().fit(x,y)
y_pred = model.predict(x)
print(y_pred)
print(round(r2_score(y,y_pred)*100,2),'%')      # 23.11 %

plt.scatter(x,y)
plt.plot(x,y_pred,c='r')
plt.show()
# 신뢰도 23.11%인 잔차가 너무 큰 모델이 나옴

# 비선형회귀 모델 적용 - 다항식 특징 추가, log 변환, curve_fit ...
from sklearn.preprocessing import PolynomialFeatures        # 다항식 특징 추가
poly = PolynomialFeatures(degree=2, include_bias=False)     # degree = 열 수, -차항 까지 만들어짐
x2 = poly.fit_transform(x)  # 특징 행렬 만들기
print(x2)


model2 = LinearRegression().fit(x2,y)
y_pred2 = model2.predict(x2)
print(y_pred2)
print(round(r2_score(y,y_pred2)*100,2),'%')      # 98.92 %

plt.scatter(x,y)
plt.plot(x,y_pred2,c='r')
plt.show()

