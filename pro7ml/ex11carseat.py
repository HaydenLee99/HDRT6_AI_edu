import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 회귀분석 문제 3)    
# kaggle.com에서 carseats.csv 파일을 다운 받아 Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.

uri = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv"
df = pd.read_csv(uri)

df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 400 entries, 0 to 399
# Data columns (total 11 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   Sales        400 non-null    float64
#  1   CompPrice    400 non-null    int64  
#  2   Income       400 non-null    int64  
#  3   Advertising  400 non-null    int64  
#  4   Population   400 non-null    int64  
#  5   Price        400 non-null    int64  
#  6   ShelveLoc    400 non-null    object 
#  7   Age          400 non-null    int64  
#  8   Education    400 non-null    int64  
#  9   Urban        400 non-null    object 
#  10  US           400 non-null    object 
# dtypes: float64(1), int64(7), object(3)

df = df.drop([df.columns[6],df.columns[9],df.columns[10]], axis=1)
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 400 entries, 0 to 399
# Data columns (total 8 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   Sales        400 non-null    float64
#  1   CompPrice    400 non-null    int64  
#  2   Income       400 non-null    int64  
#  3   Advertising  400 non-null    int64  
#  4   Population   400 non-null    int64  
#  5   Price        400 non-null    int64  
#  6   Age          400 non-null    int64  
#  7   Education    400 non-null    int64  
# dtypes: float64(1), int64(7)

print(df.corr())
#                 Sales  CompPrice    Income  Advertising  Population     Price       Age  Education
# Sales        1.000000   0.064079  0.151951     0.269507    0.050471 -0.444951 -0.231815  -0.051955
# CompPrice    0.064079   1.000000 -0.080653    -0.024199   -0.094707  0.584848 -0.100239   0.025197
# Income       0.151951  -0.080653  1.000000     0.058995   -0.007877 -0.056698 -0.004670  -0.056855
# Advertising  0.269507  -0.024199  0.058995     1.000000    0.265652  0.044537 -0.004557  -0.033594
# Population   0.050471  -0.094707 -0.007877     0.265652    1.000000 -0.012144 -0.042663  -0.106378
# Price       -0.444951   0.584848 -0.056698     0.044537   -0.012144  1.000000 -0.102177   0.011747
# Age         -0.231815  -0.100239 -0.004670    -0.004557   -0.042663 -0.102177  1.000000   0.006488
# Education   -0.051955   0.025197 -0.056855    -0.033594   -0.106378  0.011747  0.006488   1.000000

# sales와 상관성이 유의한 열로 다중선형모델 생성
lm = smf.ols(formula='Sales ~ Income+Advertising+Price+Age',data=df).fit()
print(lm.summary())

df_lm = df.iloc[:,[0,2,3,5,6]]
# print(df_lm)

# 선형회귀모델의 적절성 조건 검증 후 모델 사용
# 1. 잔차항 구하기
fitted = lm.predict(df_lm)
residual = df_lm['Sales'] - fitted
print('실제값:',df_lm['Sales'][:5].values)
print('예측값:',fitted[:5].values)
print('잔차값:',residual[:5].values)
print('잔차평균값:',np.mean(residual))

# 2. 잔차의 정규성 확인
from scipy.stats import shapiro
import statsmodels.api as sm

stat, p = shapiro(residual)
print(f"통계량:{stat:.6f}, p-value:{p:.6f}")
# 통계량:0.994922, p-value:0.212700 > 0.05
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
# 시각화는 선형성 시각화 참조

# 등분산성 검정: 독립변수의 모든 값에 대해 잔차(오차)의 분산이 일정해야 한다
# 브루시-패간(Breusch-Pagan) test
# 목적: 잔차(residual)의 분산이 일정한지 확인
# 방법: 잔차²를 독립변수에 대해 회귀 → 잔차²가 독립변수에 따라 달라지면 등분산성 위배
# 귀무 : 등분산성 만족
# 대립 : 등분산성 위배
from statsmodels.stats.diagnostic import het_breuschpagan
# sm.add_constant = 회귀에서 절편을 추정할 수 있도록 독립변수에 1로 된 열을 추가하는 함수
bp_test = het_breuschpagan(residual, sm.add_constant(df_lm['Sales']))   # (잔차, 독립변수+절편)
bp_stat, bp_pvalue = bp_test[0], bp_test[1]
print(f"breuschpagan test : 통계량={bp_stat}, p-value={bp_pvalue}")
print("등분산성 만족" if bp_pvalue>0.05 else "등분산성 위배")

# 독립성 검정 : 다중회귀 분석 시 독립변수의 값이 서로 관련되지 않아야 한다.
# 잔차가 자기상관이 있는지 확인
# Durbin-Waston : 잔차의 자기상관 검정 지표
# 시계열 데이터에서 중요함
# 값의 범위 : 0~4 , 2이면 자기상관 없음(정상), 2보다 작으면 양의 상관 2보다 크면 음의 상관
import statsmodels.api as sm
print('Durbin-Waston : ', sm.stats.stattools.durbin_watson(residual))
# Durbin-Waston :  1.9314981270829592 이므로 잔차의 자기상관은 없다.

# 다중공선성 검정 : 다중 회귀 분석시 독립변수간의 강한 상관 관계가 있으면 안된다.
# VIF(variance_inflation_factor) 분산 인플레 요인, 분산팽창지수
# 값이 10을 남으면 다중공선성이 발생하는 변수라 볼 수 있다.
from statsmodels.stats.outliers_influence import variance_inflation_factor
df_ind = df_lm.drop('Sales', axis=1)
vifdf = pd.DataFrame()
vifdf['변수'] = df_ind.columns
vifdf['vif_value'] = [variance_inflation_factor(df_ind.values, i) for i in range(df_ind.shape[1])]
print(vifdf)    # 10을 초과한 값이 없으므로 모두 다중공선성 만족하는 독립변수다.
#    vif_value
# 0   5.971040
# 1   1.993726
# 2   9.979281
# 3   8.267760

# 시각화
sns.barplot(x="변수", y="vif_value", data=vifdf)
plt.title('VIF')
plt.show()

import joblib
joblib.dump(lm, 'carseat.model')        # 모델 저장
mymodel = joblib.load('carseat.model')  # 모델 로드

new_df = pd.DataFrame({'Income':[35,62], 'Advertising':[6,3],'Price':[105,88],'Age':[32,55]})
pred = mymodel.predict(new_df)
print("Sales 예측결과", pred.values)