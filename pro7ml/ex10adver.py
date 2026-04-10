# 선형회귀분석의 모형의 적절성 선행 조건
# Advertising.csv 이용 ols 연습

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib

uri = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv"
data = pd.read_csv(uri, usecols=range(1,5))

data.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 200 entries, 0 to 199
# Data columns (total 4 columns):
#  #   Column     Non-Null Count  Dtype  
# ---  ------     --------------  -----  
#  0   tv         200 non-null    float64       (독립변수)
#  1   radio      200 non-null    float64       (독립변수)
#  2   newspaper  200 non-null    float64       (독립변수)
#  3   sales      200 non-null    float64       (종속변수)
# dtypes: float64(4)

print(data.corr())
#                  tv     radio  newspaper     sales
# tv         1.000000  0.054809   0.056648  0.782224
# radio      0.054809  1.000000   0.354104  0.576223
# newspaper  0.056648  0.354104   1.000000  0.228299
# sales      0.782224  0.576223   0.228299  1.00000

# 단순선형회귀 모델 - ols 만들기
# X : tv
# y : sales

lm = smf.ols(formula="sales ~ tv", data=data).fit()
print(lm.summary().tables[1])
print(f"coef : {lm.params}, p-value : {lm.pvalues}, r-sq : {lm.rsquared*100:.2f}%")
print("r-sq : ", round(0.782224**2*100,2),"%")  # 결정계수 = 상관계수^2
# coef : Intercept    7.032594
# tv           0.047537
# dtype: float64, p-value : Intercept    1.406300e-35
# tv           1.467390e-42
# dtype: float64, r-sq : 0.611875050850071

# 예측
x_new = pd.DataFrame({'tv':data.tv[:3]})
print(x_new)

print('실제값 :\n', data.sales[:3].values)
print('예측값 :\n', lm.predict(x_new).values)
print('예측값 직접 계산 : ',lm.params.Intercept + lm.params.tv*230.1)

# 경험하지 않은 값 예측
my_new = pd.DataFrame({'tv':[100,350,780]})
print('예측판매량', lm.predict(my_new).values)

# 시각화
plt.scatter(data.tv, data.sales)
plt.xlabel('tv광고비')
plt.ylabel('상품판매량')
y_pred = lm.predict(data.tv)
plt.plot(data.tv, y_pred, color='r')
plt.title('단순선형회귀')
plt.grid(True)
plt.show()

# 단순선형회귀모델 이므로 적절성 선행조건 중 잔차의 정규성, 선형성 확인
# 잔차(residual) : 실제값과 예측값의 차이
fitted = lm.predict(data.tv)
residual = data['sales'] - fitted
print('실제값:',data['sales'][:5].values)
print('예측값:',fitted[:5].values)
print('잔차값:',residual[:5].values)
print('잔차평균값:',np.mean(residual[:5]))

# 잔차의 정규성 확인
from scipy.stats import shapiro
import statsmodels.api as sm

stat, p = shapiro(residual)
print(f"통계량:{stat:.6f}, p-value:{p:.6f}")
# 통계량:0.990531, p-value:0.213326 > 0.05 이므로 정규성 만족
print("정규성 만족" if p > 0.05 else "정규성 위배")

# qq-plot으로 시각화
sm.qqplot(residual, line='s')
plt.title('qq-plot으로 정규성 만족 확인')
plt.show()

# 선형성 검정 : 독립변수의 변화에 따라 종속변수도 변화해야 하지만 특정한 패턴이 있으면 안됨
# 독립변수와 종속변수 간에 선형형태로 적절하게 모델링 되었는지 검정
from statsmodels.stats.diagnostic import linear_reset       # 선형성 확인 모듈

reset_result = linear_reset(lm, power=2, use_f=True)
print("reset_result 결과 : ", reset_result.pvalue)
print("선형성 만족" if reset_result.pvalue > 0.05 else "선형성 위배")

# 시각화
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()],[0,0],'--',color='grey')
plt.show()

# 등분산성 검정: 독립변수의 모든 값에 대해 잔차(오차)의 분산이 일정해야 한다
# 브루시-패간(Breusch-Pagan) test
# 목적: 잔차(residual)의 분산이 일정한지 확인
# 방법: 잔차²를 독립변수에 대해 회귀 → 잔차²가 독립변수에 따라 달라지면 등분산성 위배
# 귀무 : 등분산성 만족
# 대립 : 등분산성 위배
from statsmodels.stats.diagnostic import het_breuschpagan
# sm.add_constant = 회귀에서 절편을 추정할 수 있도록 독립변수에 1로 된 열을 추가하는 함수
bp_test = het_breuschpagan(residual, sm.add_constant(data['tv']))   # (잔차, 독립변수+절편)
bp_stat, bp_pvalue = bp_test[0], bp_test[1]
print(f"breuschpagan test : 통계량={bp_stat}, p-value={bp_pvalue}")
print("등분산성 만족" if bp_pvalue>0.05 else "등분산성 위배")

# Cook's distance : 특정 데이터가 회귀모델에 얼마나 영향을 주는지 확인 가능
# 영향력 있는 관측치(이상치) 탐지하는 진단 방법
# 데이터가 적을 때, 이상치가 의심될 때, 모델결과가 이상할 때.. 등등
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lm).cooks_distance     # Cook 거리와 idx 반환
# Cook 거리가 가장 큰 값 5개
print(cd.sort_values(ascending=False).head())
# 35     0.060494
# 178    0.056347
# 25     0.038873
# 175    0.037181
# 131    0.033895
# 영향력이 가장 큰 관측치 원본 확인
print(data.iloc[[35,178,25,175,131]])
# 대부분 tv 광고비는 매우 높으나 sales가 낮음 - 모델이 예측하기 어려운 포인트들

# 시각화
fig = sm.graphics.influence_plot(lm, alpha=0.05, criterion="cooks")
plt.show()


# 다중선형회귀 모델 - ols 만들기
# X : tv, radio, newspaper
# y : sales

lm_mul = smf.ols(formula="sales ~ tv+radio+newspaper", data=data).fit()
print(lm_mul.summary())