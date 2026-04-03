# 전통적 방법의 선형회귀(기계학습 중 지도학습에 해당)
# 각 데이터에 대한 잔차 제곱의 합이 최소가 되는 회귀선을 만들고,
# 이를 통해 독립변수가 종속 변수에 얼마나 영향을 주는지 인과관계를 분석
# 독립변수: 연속형
# 종속변수: 연속형
# 두 변수는 상관관계 및 인과관계가 있어야 함.
# 정량적인 모델을 생성

import numpy as np
import statsmodels.api as sm
from sklearn.datasets import make_regression

np.random.seed(12)

# 방법 1 : make_regression 사용. 모델 생성 X
x,y,coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True)
print(x)
print(y)
print(coef)

# 방법 2 : Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
fit_model = model.fit(x,y)
print('slope',fit_model.coef_)
print('bias',fit_model.intercept_)

# 예측값 확인 함수
y_new1 = fit_model.predict(x[[0]])
y_new2 = fit_model.predict([[0],[0.12345]])   # 2차원 값으로 입력
print(y_new1,y_new2)


# 방법 3 : ols 사용
# 잔차제곱합(rss)을 최소화 하는 가중치 벡터를 행렬 미분으로 구하는 방법
import statsmodels.formula.api as smf
x1 = x.flatten()
print(x1.ndim)
y1 = y
data = np.array([x1,y1])

import pandas as pd
df=pd.DataFrame(data.T)
df.columns=['x1','y1']
print(df)

model2 = smf.ols(formula="y1 ~ x1",data=df).fit()
print(model2.summary())
print(model2.params['x1'])      # slope
print(model2.params['Intercept'])       # bias

# 예측값 확인
new_df = pd.DataFrame({'x1':[-1.70073563,-0.67794537]}) #기존자료검증
print(model2.predict(new_df))

new_df2 = pd.DataFrame({'x1':[0.12345,0.2345]}) #새로운 데이터
print(model2.predict(new_df2))

