# ANOVA (Analysis of Variance, 분산분석)
# 세 개 이상의 집단 평균이 서로 차이가 있는지를 검정하는 통계 기법

# 귀무 : 모든 집단 평균이 같다.
# 대립 : 적어도 한 집단 평균은 다르다.

# 검정 결과 해석:
#  p-value > 유의수준  → 귀무가설 채택  → 모든 평균이 같다
#  p-value ≤ 유의수준  → 귀무가설 기각  → 적어도 한 평균이 다르다

# 오류 유형:
# 1종 오류 (Type I error, α 오류)
# - 귀무가설이 참인데 기각

# 2종 오류 (Type II error, β 오류)
# - 귀무가설이 거짓인데 채택

# 3종 오류 (Type III error)
# - 통계적으로 차이가 있다고 나오지만, 실제 의미는 무시할 정도임

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols             # 추정 및 검정, 회귀, 시계열 분석 등 기능 제공
# ols (Ordinart Least Squares - 최소제곱법) : 절편과 기울기를 구해, 회귀모델을 구할 수 있음
import statsmodels.api as sm
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sqlalchemy import create_engine

pd.set_option('display.max_columns', None)


# 실습1 - One way ANOVA
# 세가지 교육 방법을 적용하여 1개월 동안 교육받은 교육생 8명을 대상으로 실기 시험 실시한 데이터 three_sample.csv

# 독립변수(범주형) : 교육방법(1요인)-(3가지 방법)
# 종속변수(연속형) : 실기시험 평균점수

# 귀무 : 세 가지 교육 방법에 따흔  시험점수의 차이가 있다.
# 대립 : 세 가지 교육 방법에 따흔  시험점수의 차이가 없다.
url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/three_sample.csv"
df = pd.read_csv(url)
# df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 80 entries, 0 to 79              <- 80건 데이터
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   no      80 non-null     int64
#  1   method  80 non-null     int64
#  2   survey  80 non-null     int64
#  3   score   80 non-null     int64
# dtypes: int64(4)
# memory usage: 2.6 KB

# print(df.describe())
#             no     method     survey       score
# count  80.0000  80.000000  80.000000   80.000000
# mean   40.5000   1.962500   0.650000   78.212500
# std    23.2379   0.802587   0.479979   64.886404
# min     1.0000   1.000000   0.000000   33.000000
# 25%    20.7500   1.000000   0.000000   58.000000
# 50%    40.5000   2.000000   1.000000   65.000000
# 75%    60.2500   3.000000   1.000000   79.500000
# max    80.0000   3.000000   1.000000  500.000000  <- 시험점수 평균이 500인 이상치 존재

# 이상치 시각화
# sns.boxplot(df.score)
# plt.show()

data = df.query("0 <= score <= 100")
# print(len(data))        # 78건 , 이상치 2건 제거

# 시각화
# sns.boxplot(data.score)
# plt.show()



# 교차표 이용

# 교육 방법별 건수
data2 = pd.crosstab(index=data['method'], columns='count')
data2.index = ['방법1','방법2','방법3']
# print(data2)

# 교육 방법별 만족 건수
data3 = pd.crosstab(data['method'], data['survey'])
data3.index = ['방법1','방법2','방법3']
data3.columns = ['만족','불만족']
# print(data3)

print("ANOVA 검정---")
# F 통계값을 얻기 위해 회귀분석 결과(선형모델)를 사용
linreg = ols("data['score'] ~ data['method']", data=data).fit()     # 회귀분석 모델 생성
result = sm.stats.anova_lm(linreg, type=1)
print(result)

# 교육방법에 따라 시험점수에 차이여부는 알 수 있지만
#  정확히 어느 그룹의 평균값이 의미있는지는 알 수 없다
# 그러므로 추가적인 사후분석을 필수적으로 시행
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukResult = pairwise_tukeyhsd(endog = data['score'], groups=(data['method']))
print(tukResult)

# 사후분석 시각화
tukResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
# Tukey HSD : 원래 반복수가 동일하다는 가정하에 고안된 방법
# 집단간 평균차이 출력 가능

f_value = result.loc["data['method']", "F"]
p_value = result.loc["data['method']", "PR(>F)"]