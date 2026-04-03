import numpy as np
import pandas as pd
from scipy.stats import linregress                  # 단순 선형회귀용
from statsmodels.formula.api import ols             # 통계 기반 회귀용
from sklearn.linear_model import LinearRegression   # 머신러닝 스타일 회귀
from sklearn.metrics import mean_squared_error

# 회귀분석 문제 1) scipy.stats.linregress(), statsmodels ols(), LinearRegression 사용
# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
data = {
    '구분': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    '지상파': [0.9,1.2,1.2,1.9,3.3,4.1,5.8,2.8,3.8,4.8,np.nan,0.9,3.0,2.2,2.0],
    '종편': [0.7,1.0,1.3,2.0,3.9,3.9,4.1,2.1,3.1,3.1,3.5,0.7,2.0,1.5,2.0],
    '운동': [4.2,3.8,3.5,4.0,2.5,2.0,1.3,2.4,1.3,35.0,4.0,4.2,1.8,3.5,3.5]
}
#  지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#  지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
# 결측치는 해당 칼럼의 평균 값을 사용. 
# 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치.  

# 주어진 데이터 전처리
data = pd.DataFrame(data)
data['지상파'] = data['지상파'].fillna(data['지상파'].mean())
# print(data.isnull().sum())
data = data[data['운동'] <= 10]
# print(data)


# linregress
model = linregress(data['지상파'], data['운동'])
# print([model.slope, model.intercept])
print('점수예측 : ',np.polyval([model.slope, model.intercept], data['지상파']))

model = linregress(data['지상파'], data['종편'])
# print([model.slope, model.intercept])
print('점수예측 : ',np.polyval([model.slope, model.intercept], data['지상파']))


# Linear Regression
X = data['지상파'].values.reshape(-1,1)
y = data['운동'].values.reshape(-1,1)

model = LinearRegression()
fit_model = model.fit(X, y)
y_new = fit_model.predict(X)
print(y_new)

y = data['종편'].values.reshape(-1,1)
fit_model = model.fit(X, y)
y_new = fit_model.predict(X)
print(y_new)


# ols 사용
X = data['지상파']
y = data['운동']
df_ols = pd.DataFrame({'X': X, 'y': y})

model = ols(formula="y ~ X",data=df_ols).fit()
print(model.predict(df_ols))

y = data['종편']
df_ols = pd.DataFrame({'X': X, 'y': y})

model = ols(formula="y ~ X",data=df_ols).fit()
print(model.predict(df_ols))