import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sqlalchemy import create_engine

# [ANOVA 예제 1]
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.

# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.
# 수집된 자료 :  
data = {
    'kind': [1, 2, 3, 4, 2, 1, 3, 4, 2, 1, 2, 3, 4, 1, 2, 1, 1, 3, 4, 2],
    'quantity': [64, 72, 68, 77, 56, np.nan, 95, 78, 55, 91, 63, 49, 70, 80, 90, 33, 44, 55, 66, 77]
}

# 귀무 : 빵을 기름에 튀길 때 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 없다.
# 대립 : 빵을 기름에 튀길 때 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 있다.

dfq1 = pd.DataFrame(data=data)
dfq1['quantity'] = dfq1.groupby('kind')['quantity'].transform(lambda x: x.fillna(x.mean()))

gr1 = dfq1[dfq1.kind == 1]['quantity'].to_numpy()
gr2 = dfq1[dfq1.kind == 2]['quantity'].to_numpy()
gr3 = dfq1[dfq1.kind == 3]['quantity'].to_numpy()
gr4 = dfq1[dfq1.kind == 4]['quantity'].to_numpy()

# print(gr1)      # [64.  62.4 91.  80.  33.  44. ]
# print(gr2)      # [72. 56. 55. 63. 90. 77.]
# print(gr3)      # [68. 95. 49. 55.]
# print(gr4)      # [77. 78. 70. 66.]

if  stats.shapiro(gr1).pvalue < 0.05 or \
    stats.shapiro(gr2).pvalue < 0.05 or \
    stats.shapiro(gr3).pvalue < 0.05 or \
    stats.shapiro(gr4).pvalue < 0.05:
    shapiro_msg = False
else:
    shapiro_msg = True

if stats.bartlett(gr1,gr2,gr3,gr4).pvalue < 0.05:
    bartlett_msg = False
else:
    bartlett_msg = True

if shapiro_msg and bartlett_msg:
    print("일원분산분석(One-Way ANOVA)...")
    if stats.f_oneway(gr1,gr2,gr3,gr4).pvalue < 0.05:
        print(f"p-value {stats.f_oneway(gr1,gr2,gr3,gr4).pvalue} < 0.05 이므로 귀무 기각")
    else:
        print(f"p-value {stats.f_oneway(gr1,gr2,gr3,gr4).pvalue} > 0.05 이므로 귀무 채택")

# 사후검정
tukResult = pairwise_tukeyhsd(endog=dfq1.quantity, groups=dfq1.kind, alpha=0.05)
print(tukResult)

# 시각화
tukResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()

# 일원분산분석(One-Way ANOVA) 결과 해석
# p-value = 0.809 > 0.05 이므로 귀무가설 채택
# 빵을 기름에 튀길 때 기름의 종류에 따라 흡수하는 기름의 평균 quantity 차이는 통계적으로 유의하지 않음

# 사후검정(Tukey HSD) 결과 해석
# 모든 그룹 간 비교에서 reject=False
# 각 그룹 평균 간 차이는 통계적으로 유의하지 않음
# 즉, 어떤 그룹끼리 비교해도 평균량의 차이는 신뢰할 수 있는 수준에서 없다고 판단


# [ANOVA 예제 2]
# DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오. 
# 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.

# 귀무 : 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 없다.
# 대립 : 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있다.

sql = """
SELECT 
    j. jikwonno AS 사번,
    b.busername AS 부서명,
    j.jikwonpay AS 연봉
FROM jikwon j
INNER JOIN buser b ON j.busernum = b.buserno
WHERE b.busername IN ('총무부', '영업부', '관리부', '전산부')
    AND j.jikwonpay IS NOT NULL;
"""

engine = create_engine("mysql+pymysql://root:123@localhost:3306/test?charset=utf8mb4")
data = pd.read_sql(sql, engine)
data.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 30 entries, 0 to 29
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   사번      30 non-null     int64
#  1   부서명     30 non-null     object
#  2   연봉      30 non-null     int64
# dtypes: int64(2), object(1)
# memory usage: 852.0+ bytes

gr1 = data[data.부서명 == '총무부']['연봉'].to_numpy()
gr2 = data[data.부서명 == '영업부']['연봉'].to_numpy()
gr3 = data[data.부서명 == '관리부']['연봉'].to_numpy()
gr4 = data[data.부서명 == '전산부']['연봉'].to_numpy()

# print(gr1)      # [9900 3700 4900 8000 3800 3500 4100]
# print(gr2)      # [8800 7900 3000 2950 7800 4000 3000 2900 2950 4500 5900 5200]
# print(gr3)      # [8600 7200 3400 5850]
# print(gr4)      # [4500 5000 3900 7800 5500 6600 4000]

if  stats.shapiro(gr1).pvalue < 0.05 or \
    stats.shapiro(gr2).pvalue < 0.05 or \
    stats.shapiro(gr3).pvalue < 0.05 or \
    stats.shapiro(gr4).pvalue < 0.05:
    shapiro_msg = False
else:
    shapiro_msg = True

if stats.bartlett(gr1,gr2,gr3,gr4).pvalue < 0.05:
    bartlett_msg = False
else:
    bartlett_msg = True

# print("kruskal...")
# print(stats.kruskal(gr1,gr2,gr3,gr4).pvalue)
# 0.643 > 0.05 이므로 귀무가설 채택

# 사후검정
tukResult = pairwise_tukeyhsd(endog=data.연봉, groups=data.부서명, alpha=0.05)
print(tukResult)

# 시각화
tukResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()

# - 총무부, 영업부, 전산부, 관리부 직원들의 평균 연봉 차이는 신뢰할 수 있는 수준에서 없다
# - Tukey HSD 시각화도 모든 그룹이 겹치는 범위를 가지므로 차이가 없음을 시각적으로 확인 가능