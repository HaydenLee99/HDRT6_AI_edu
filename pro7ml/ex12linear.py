# Linear Regression 클래스 사용 : 평가 score 정리

import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler      # 정규화 클래스

# 데이터 생성
sample_size = 100
np.random.seed(1)

x = np.random.normal(0,10,sample_size)
y = np.random.normal(0,10,sample_size) + x*30
print(x[:5],y[:5])
print('상관계수:',np.corrcoef(x,y)[0,1])

# 독립변수 x 정규화(0~1)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1,1))
print(x_scaled[:5])

# 시각화
plt.scatter(x_scaled, y)
plt.show()

# 모델 생성
model = LinearRegression().fit(x_scaled,y)
print('model:', model)
print('회귀계수(slope):',model.coef_)
print('회귀계수(intercept, bias):',model.intercept_)
print('결정계수(R^2):',model.score(x_scaled,y))

y_pred = model.predict(x_scaled)
print('예측값 : ', y_pred[:5])
print('실제값 : ', y[:5])

# 모델 성능 확인 함수 작성
def myRegScoreFunc(y_ture, y_pred):
    # 결정계수 : 실제 관측값의 분산대비 예측값의 분산을 계산하여 데이터 예측의 정확도 성능 측정 지표
    print(f"R2 Score(결정계수):{r2_score(y_ture, y_pred)}")
    # 모델이 데이터의 분산을 얼마나 잘 설명하는지 나타내는 지표
    print(f"explained_variance_score(설명분산점수):{explained_variance_score(y_ture, y_pred)}")
    # 오차를 제곱해 평균을 구함
    print(f"mean_squared_error(MSE, 평균제곱오차):{mean_squared_error(y_ture, y_pred)}")

myRegScoreFunc(y, y_pred)

# 분산이 크게 다른 x,y 값 사용
x2 = np.random.normal(0,1,sample_size)
y2 = np.random.normal(0,100,sample_size) + x*30
print(x2[:5],y2[:5])
print('상관계수:',np.corrcoef(x2,y2)[0,1])

# 독립변수 x 정규화(0~1)
x2_scaled = scaler.fit_transform(x2.reshape(-1,1))
print(x2_scaled[:5])

# 시각화
plt.scatter(x2_scaled, y2)
plt.show()

# 모델 생성
model2 = LinearRegression().fit(x2_scaled,y2)
print('model:', model2)
print('회귀계수(slope):',model2.coef_)
print('회귀계수(intercept, bias):',model2.intercept_)
print('결정계수(R^2):',model2.score(x2_scaled,y2))

y2_pred = model2.predict(x2_scaled)
print('예측값 : ', y2_pred[:5])
print('실제값 : ', y2[:5])
# 분산이 너무 다른 데이터로 만든 모델은 의미가 없다