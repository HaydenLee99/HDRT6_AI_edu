# 참고 : 70년대 미국 보스턴 시의 주택가격을 설명한 dataset
# 회귀분석의 한 예로 scikit-learn 패키지에서 제공하는 주택가격을 예측하는 Dataset을 사용할 수 있다. 
# 이는 범죄율, 공기 오염도 등의 주거 환경 정보 등을 사용하여 70년대 미국 보스턴 시의 주택가격을 표시하고 있다.

# * 데이터 세트 특성 :
#     : 인스턴스 수 : 506
#     : 속성의 수 : 13 개의 숫자 / 범주 적 예측
#     : 중간 값 (속성 14)은 대개 대상입니다
#     : 속성 정보 (순서대로) :

# CRIM   자치시(town) 별 1인당 범죄율
# ZN 25,000   평방피트를 초과하는 거주지역의 비율
# INDUS   비소매상업지역이 점유하고 있는 토지의 비율
# CHAS   찰스강에 대한 더미변수(강의 경계에 위치한 경우는 1, 아니면 0)
# NOX   10ppm 당 농축 일산화질소
# RM   주택 1가구당 평균 방의 개수
# AGE   1940년 이전에 건축된 소유주택의 비율
# DIS   5개의 보스턴 직업센터까지의 접근성 지수
# RAD   방사형 도로까지의 접근성 지수
# TAX   10,000 달러 당 재산세율
# PTRATIO   자치시(town)별 학생/교사 비율
# B   1000(Bk-0.63)^2, 여기서 Bk는 자치시별 흑인의 비율을 말함.
# LSTAT   모집단의 하위계층의 비율(%)
# MEDV   본인 소유의 주택가격(중앙값) (단위: $1,000)

# ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# http://archive.ics.uci.edu/ml/datasets/Housing
# 보스톤 주택 가격 데이터는 회귀를 다루는 많은 기계 학습 논문에서 사용되었다

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

uri = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/housing.data"
data = pd.read_csv(uri, header=None, sep=r'\s+')        # sep=r'\s+' : 공백 여러 개를 구분자로 처리
data.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
# print(data.head(3))
#       CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT  MEDV
# 0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296.0     15.3  396.90   4.98  24.0
# 1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242.0     17.8  396.90   9.14  21.6
# 2  0.02729   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242.0     17.8  392.83   4.03  34.7

print(np.corrcoef(data.LSTAT, data.MEDV)[0,1])      # -0.7376  음의 상관관계를 가짐
x=data[['LSTAT']].values
y=data['MEDV']
print(x[:3],'\n',y[:3])

# 단항 선형회귀 모델
model=LinearRegression()

# 다항 특성
quad = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

x_quad = quad.fit_transform(x)
x_cubic = cubic.fit_transform(x)
# print(x_quad[:3],'\n',x_cubic[:3])

# 단순 회귀
model.fit(x,y)
x_fit = np.linspace(x.min(), x.max(), 100)[:, np.newaxis]    # 그래프 표시용
y_lin_fit = model.predict(x_fit)
# print('y_lin_fit : ', y_lin_fit)
model_r2 = r2_score(y,model.predict(x))
print('model_r2 : ', model_r2)          # 0.5441

# 2차
model.fit(x_quad,y)
y_quad_fit = model.predict(quad.fit_transform(x_fit))
# print('y_quad_fit : ', y_quad_fit)
quad_r2 = r2_score(y,model.predict(x_quad))
print('quad_r2 : ', quad_r2)            # 0.6407

# 3차
model.fit(x_cubic,y)
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))
# print('y_cubic_fit : ', y_cubic_fit)
cubic_r2 = r2_score(y,model.predict(x_cubic))
print('cubic_r2 : ', cubic_r2)          # 0.6578

plt.scatter(x,y, label='초기 데이터', c='grey')
plt.plot(x_fit, y_lin_fit, linestyle=':', label='linear fit(d=1), $R^2=%.2f$'%model_r2, c='b', lw=3)
plt.plot(x_fit, y_quad_fit, linestyle='-', label='quad fit(d=2), $R^2=%.2f$'%quad_r2, c='r', lw=3)
plt.plot(x_fit, y_cubic_fit, linestyle='--', label='cubic fit(d=3), $R^2=%.2f$'%cubic_r2, c='k', lw=3)
plt.xlabel('하위계층 비율')
plt.ylabel('주택가격 중위값')
plt.legend()
plt.show()