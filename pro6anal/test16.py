# 어느 음식점의 매출 데이터와 기상청이 제공한 날씨 데이터를 활용하여 강수 여부에 따른 매출액의 평균에 차이가 있는지 검정
# 최고 온도에 따른 매출액의 평균에 차이가 있는지 검정
# 세 집단 : 추움, 보통, 더움

# 귀무 : 어느 음식점의 매출 데이터는 온도에 따라 매출액 평균에 차이가 없다.
# 대립 : 어느 음식점의 매출 데이터는 온도에 따라 매출액 평균에 차이가 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

np.set_printoptions(suppress=True, precision=10)
pd.set_option('display.float_format', '{:.10f}'.format)
pd.set_option("display.max_columns", None)

# 매출 데이터 읽기
sales_url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv"
sales_data = pd.read_csv(sales_url, dtype={'YMD':'object'})
# print(sales_data.info())      # 328 X 3

# 기상 데이터 읽기
weather_url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv"
weather_data = pd.read_csv(weather_url)
# print(weather_data.info())    # 702 X 9

# 두 데이터의 날짜 form(YYYYMMdd)을 일치시켜 준 후 두 데이터를 병합하자.
weather_data.tm = weather_data.tm.map(lambda x:x.replace('-',''))
# print(weather_data.head())

#    병합 merge
frame = sales_data[['YMD','AMT']].merge(weather_data[['tm','maxTa']], how='left', left_on='YMD', right_on='tm')
data = frame.drop('tm', axis=1)
# print(data.head())
#         YMD     AMT  maxTa
# 0  20190514       0   26.9
# 1  20190519   18000   21.6
# 2  20190521   50000   23.8
# 3  20190522  125000   26.5
# 4  20190523  222500   29.2
# print(data.isnull().sum())                # 0

# print(data.describe())
# plt.boxplot(data.maxTa)
# plt.show()

# 온도를 3 그룹으로 분리 (연속형(int) -> 범주형(category))
print(data.isnull().sum())
data['ta_category'] = pd.cut(data.maxTa, bins=[-5, 8, 24, 37], labels=['추움', '보통', '더움'])
print(data.head(3),' ',data['ta_category'].unique())

# 정규성 & 등분산성 확인
gr1 = np.array(data[data.ta_category == '추움'].AMT)
gr2 = np.array(data[data.ta_category == '보통'].AMT)
gr3 = np.array(data[data.ta_category == '더움'].AMT)

print(stats.shapiro(gr1).pvalue)               # 0.24819 > 0.05        
print(stats.shapiro(gr2).pvalue)               # 0.03882 < 0.05        정규성 위배
print(stats.shapiro(gr3).pvalue)               # 0.31829 > 0.05        
print(stats.levene(gr1,gr2,gr3).pvalue)        # 0.03900 < 0.05        등분산성 불만족
print(stats.bartlett(gr1,gr2,gr3).pvalue)      # 0.00967 < 0.05        

print()
# 온도별 매출액 평균
spp = data.loc[:, ['AMT', 'ta_category']]
print(spp.groupby('ta_category').mean())

# plt.boxplot([gr1,gr2,gr3], showmeans=True)
# plt.show()

print(stats.kruskal(gr1,gr2,gr3))
# statistic=132.70, pvalue=1.527e-29
# 해석 : p-value가 < 0.05 이므로 귀무 기각

# ANOVA 선택
# 정규성 (Normality) & 등분산성 (Homoscedasticity)에 따라 검정 방법 선택

#  정규성    │   등분산성   │  사용 검정          
# ──────────┼────────────┼──────────────────────
#    O      │    O       │ 일반 ANOVA (f_oneway)
#    O      │    X       │ Welch ANOVA          
#    X      │    O       │ Kruskal-Wallis       
#    X      │    X       │ Kruskal-Wallis       

# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukResult = pairwise_tukeyhsd(endog=spp.AMT, groups=spp.ta_category, alpha=0.05)
print(tukResult)

# 시각화
tukResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()

# 온도에 따른 세 집단(더움, 보통, 추움) 간 모든 쌍에서 평균 차이가 통계적으로 유의하다.
# 즉, 온도 조건에 따라 AMT 평균은 모두 서로 다르다.
# (더움 > 보통 > 추움)