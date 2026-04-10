# Linear Regression 클래스 사용 : 평가 score - mtcars dataset 사용
import statsmodels.api
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data

mtcars.info()
# <class 'pandas.core.frame.DataFrame'>
# Index: 32 entries, Mazda RX4 to Volvo 142E
# Data columns (total 11 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   mpg     32 non-null     float64
#  1   cyl     32 non-null     int64  
#  2   disp    32 non-null     float64
#  3   hp      32 non-null     int64  
#  4   drat    32 non-null     float64
#  5   wt      32 non-null     float64
#  6   qsec    32 non-null     float64
#  7   vs      32 non-null     int64  
#  8   am      32 non-null     int64  
#  9   gear    32 non-null     int64  
#  10  carb    32 non-null     int64  
# dtypes: float64(5), int64(6)

print('상관계수\n', mtcars.corr(method='pearson'))
#             mpg       cyl      disp        hp      drat        wt      qsec        vs        am      gear      carb
# mpg   1.000000 -0.852162 -0.847551 -0.776168  0.681172 -0.867659  0.418684  0.664039  0.599832  0.480285 -0.550925
# cyl  -0.852162  1.000000  0.902033  0.832447 -0.699938  0.782496 -0.591242 -0.810812 -0.522607 -0.492687  0.526988
# disp -0.847551  0.902033  1.000000  0.790949 -0.710214  0.887980 -0.433698 -0.710416 -0.591227 -0.555569  0.394977
# hp   -0.776168  0.832447  0.790949  1.000000 -0.448759  0.658748 -0.708223 -0.723097 -0.243204 -0.125704  0.749812
# drat  0.681172 -0.699938 -0.710214 -0.448759  1.000000 -0.712441  0.091205  0.440278  0.712711  0.699610 -0.090790
# wt   -0.867659  0.782496  0.887980  0.658748 -0.712441  1.000000 -0.174716 -0.554916 -0.692495 -0.583287  0.427606
# qsec  0.418684 -0.591242 -0.433698 -0.708223  0.091205 -0.174716  1.000000  0.744535 -0.229861 -0.212682 -0.656249
# vs    0.664039 -0.810812 -0.710416 -0.723097  0.440278 -0.554916  0.744535  1.000000  0.168345  0.206023 -0.569607
# am    0.599832 -0.522607 -0.591227 -0.243204  0.712711 -0.692495 -0.229861  0.168345  1.000000  0.794059  0.057534
# gear  0.480285 -0.492687 -0.555569 -0.125704  0.699610 -0.583287 -0.212682  0.206023  0.794059  1.000000  0.274073
# carb -0.550925  0.526988  0.394977  0.749812 -0.090790  0.427606 -0.656249 -0.569607  0.057534  0.274073  1.000000

# hp가 mpg에 영향을 주는 인과관계
x = mtcars[['hp']].values   # 2차원
print(x[:5])
y = mtcars['mpg'].values    # 1차원
print(y[:5])

# 모델 생성
lmodel = LinearRegression().fit(x,y)
print('slope : ', lmodel.coef_)
print('intercept : ', lmodel.intercept_)

plt.scatter(x,y)
plt.plot(x, x*lmodel.coef_ + lmodel.intercept_, c='r')
plt.show()

# mpg를 예측
pred = lmodel.predict(x)

print('예측값 : ', np.round(pred[:5],1))
print('실제값 : ', y[:5])

# 모델 성능 지표
# MSE : 손실함수로 많이 사용, 계산 편리 (단위: 제곱값)
# RMSE : 해석/보고용, 직관적 (단위: 원래 값)
# MSE, RMSE는 데이터 스케일에 따라 값이 달라짐
# (단, R²는 0~1 범위, 경우에 따라 음수 가능)
# 따라서 같은 데이터셋 기준에서 모델 간 상대 비교를 한다.
print('MSE : ',mean_squared_error(y, pred))                # MSE :  13.989822298268805
print('RMSE : ',np.sqrt(mean_squared_error(y, pred)))      # RMSE :  3.7402970868994894
print('R2_score : ',r2_score(y,pred))                      # R2_score :  0.602437341423934
# R² : 설명력 지표, 변수 추가 시 증가하는 경향 → Adjusted R²로 보완
# 따라서 모델 평가는 R²(설명력) + RMSE/MSE(오차)를 함께 고려

# 새로운 hp로 mpg 예측
new_hp = [[100],[110],[120],[130]]
new_pred = lmodel.predict(new_hp)
print('예측결과 : ', new_pred.flatten())

