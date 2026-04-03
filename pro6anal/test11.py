# 어느 음식점의 매출 데이터와 기상청이 제공한 날씨 데이터를 활용하여 강수 여부에 따른 매출액의 평균에 차이가 있는지 검정
# 두 집단 : 강수량이 있을 때, 맑을 때

# 귀무 : 어느 음식점의 매출 데이터는 강수 여부에 따라 매출액 평균에 차이가 없다.
# 대립 : 어느 음식점의 매출 데이터는 강수 여부에 따라 매출액 평균에 차이가 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

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
frame = sales_data[['YMD','AMT']].merge(weather_data[['tm','maxTa','sumRn']], how='left', left_on='YMD', right_on='tm')
# print(frame.head(),' 총',len(frame),'건')   # 328 건
data = frame.drop('tm', axis=1)
# print(data.head())
# print(data.isnull().sum())                # 0

# 독립 표본 t-test 시작

# 강수량 있으면 1, 없으면 0인 컬럼 추가
data['rain_yn'] = (data['sumRn'] > 0).astype(int)
# print(data.head())

# box-plot으로 시각화
sp = np.array(data.iloc[:,[1,4]])
# print(sp[:3])               # [AMT, rain_yn] 꼴

tg1 = sp[sp[:,1]==0, 0]     # 집단 1 : 맑은 날 매출
tg2 = sp[sp[:,1]==1, 0]     # 집단 2 : 비온 날 매출
# print(tg1[:3])
# print(tg2[:3])

# print('맑은 날 매출액 평균 : ', np.mean(tg1).round(2))
# 맑은 날 매출액 평균 :  761040.25
# print('비온 날 매출액 평균 : ', np.mean(tg2).round(2))
# 비온 날 매출액 평균 :  757331.52

plt.boxplot([tg1, tg2], meanline=True, showmeans=True, notch=True)
plt.show()

# 정규성 검정
print(len(tg1), ' ', len(tg2))      # 236   92
print(stats.shapiro(tg1).pvalue)    # 0.0560506 > 0.05 만족
print(stats.shapiro(tg2).pvalue)    # 0.8827503 > 0.05

# 등분산 검정
print(stats.levene(tg1, tg2).pvalue)    # 0.7123452 > 0.05 만족

print(stats.ttest_ind(tg1, tg2, equal_var=True))
# statistic 0.1010982, pvalue 0.9195345, df 326
# 해석 : 정규성, 등분산성 조건은 충족함.
# pvalue 0.9195345 > alpha 0.05 이므로 귀무가설 채택
# 매출 데이터는 강수 여부에 따라 매출액 평균에 차이가 없다고 보여진다.

