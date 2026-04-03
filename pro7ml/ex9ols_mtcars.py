# 선형회귀분석 : mtcars dataset
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api
import matplotlib.pyplot as plt
import koreanize_matplotlib

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head())
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

# 마력이 연비에 영향을 주는가?
# x: hp         y: mpg
print(mtcars[['wt','hp','mpg']].corr())
#            wt        hp       mpg
# wt   1.000000  0.658748 -0.867659
# hp   0.658748  1.000000 -0.776168
# mpg -0.867659 -0.776168  1.000000
# 연비는 마력, 차체 무게와 강한 음의 상관 관계를 가짐

# 시각화
# plt.scatter(mtcars.hp, mtcars.mpg)
# plt.xlabel('마력')
# plt.ylabel('연비')
# plt.show()

# 단순선형회귀
result = smf.ols(formula='mpg ~ hp', data=mtcars).fit()
print(result.summary())
print('R-squared : ', result.rsquared)          # 0.602
print('p-values : ', result.pvalues.iloc[1])    # 1.787e-07
# mpg_hat = -0.0682 * hp + 30.0989 + err
print('마력수 110에 대한 연비 예측값 : ', result.predict(pd.DataFrame({'hp':[110]})))

# 다중선형회귀
result2 = smf.ols(formula='mpg ~ hp + wt',data=mtcars).fit()
print(result2.summary())
print('adj. R-squared : ', result.rsquared_adj)          # 0.815
print('마력수 110, 차체무게 5에 대한 연비 예측값 : ', result2.predict(pd.DataFrame({'hp':[110], 'wt':[5]})))

# 단순선형회귀
result3 = smf.ols(formula='mpg ~ wt',data=mtcars).fit()
print(result3.summary())
print('결정계수 : ', result.rsquared)          # 0.815
print('result3 연비 예측값 : ', result3.predict()[:5])

# 새로운 차체 무게로 연비 추정
mtcars.wt = float(input('차체 무게 입력: '))
new_pred = result3.predict(pd.DataFrame(mtcars.wt))
print(f"차체무게 {mtcars.wt[0]}일 때 예상 연비는 {new_pred[0]}")