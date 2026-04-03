# 단일 모집단의 평균에 대한 가설검정
# 실습2 : A 중학교 국어점수 평균검정(80)
# 귀무  : 국어점수 평균은 <80점> 이다.
# 대립  : 국어점수 평균은 <80점> 이 아니다.
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv")
print(data.describe())
print(len(data['국어']))         # 20 : 30행이 넘으면 중심극한 정리의 의해 정규성을 따른다고 가정.
                                # 30개가 넘지 않으므로 정규성 검정 필요
# shapiro-wilk test : 데이터가 정규분포를 따르는지 여부를 통계적으로 확인하는 검정
# 귀무 : 데이터가 <정규 분포> 이다. 정규성을 따른다.
print(stats.shapiro(data['국어']))
# statistic=0.8724, pvalue=0.0129
# 해석 : 유의 수준 0.05 > pvalue 귀무가설 기각 -> 정규성 위배

# 정규성을 만족하지 못한 경우 대안 : 비모수 검정 방법 사용
# 윌콕슨 검정(Wilcoxon signed-rank test) 
# : 정규성을 가정하지 않고, 같은 집단의 전/후 또는 짝을 이루는 두 관련 샘플의 중앙값 차이가 유의한지 검정하는 비모수 통계 방법
# 귀무 : 두 관련 샘플의 중앙값(median) 차이가 없다. (중앙값 차이 0 이다.)
result = stats.wilcoxon(data['국어'] - 80)
print(result)
# statistic=74.0, pvalue=0.3977
# 해석 : 유의 수준 0.05 < pvalue=0.3977 귀무가설 채택

result_t = stats.ttest_1samp(data['국어'], popmean=80)
print(result_t)
# statistic=-1.3321, pvalue=0.19856, df=19
# 해석 : 유의 수준 0.05 < pvalue=0.19856 귀무가설 채택


# 결론 :
# 정규성은 부족하나 귀무가설 채택이라는 동일 결론을 얻음
# 표본 수가 충분히 크다면 t-검정 사용 가능

# 보고서 작성시
# shapiro-wilk test 결과 정규성 가정이 다소 위배되었으나
# wilcoxon 결과도 동일하므로 t-검정 결과를 신뢰할 수 있다 라고 명시한다.

print('-'*100)
# 실습2
# 여아신생아 몸무게의 평균검정수행
# popmean = 2800(g)
# 이보다 크다는 주장이 나왔다
# 표본 여아 18명을 뽑아 체중을 측정한 데이터를 이용하여 주장이 맞는지 검증해보자

# 귀무 : 여아 신생아의 몸무게는 <평균 2800(g)> 이다
# 대립 : 여아 신생아의 몸무게는 <평균 2800(g)> 보다 크다

data2 = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/babyboom.csv")
print(data2.head())     # gender 1이 여아
fdata = data2[data2.gender == 1]
print(f"여아 데이터 건수 : {len(fdata)}건")     # 18건

# 표본수가 18 (< 30) 이므로 정규성 검증 필요
shapiro_result2 = stats.shapiro(fdata['weight'])
print(shapiro_result2)
# statistic=0.8702, pvalue=0.0179
# 해석 : 유의 수준 0.05 > pvalue 0.0179 이므로 귀무가설 기각 -> 정규성 위배

# 정규성 만족여부 시각화 1
sns.histplot(fdata['weight'], kde=True)
plt.show()

# 정규성 만족여부 시각화 2 - Quantile-Quantile plot : 잔차가 커브를 그리면 정규성 만족 못함
stats.probplot(fdata['weight'], plot=plt)
plt.show()

# 정규성을 만족하지 못하므로 윌콕슨 검정(Wilcoxon signed-rank test) 사용
wilconxon_result2 = stats.wilcoxon(fdata['weight'] - 2800)
print(wilconxon_result2)
# statistic=37.0, pvalue=0.0342
# 해석 : 유의 수준 0.05 > pvalue=0.0342 귀무가설 기각

# one sample t-test
t_result2 = stats.ttest_1samp(fdata['weight'], popmean=2800)
print(t_result2)
# statistic=2.2331, pvalue=0.03926, df=17
# 해석 : 유의 수준 0.05 > pvalue=0.03926 이므로 귀무가설 기각
# 해석 : df 17, 유의수준 0.05 이므로 t-분포표를 보면 임계값(cv)는 1.740 이다. t값 2.2331이 임계값 보다 크므로 귀무가설 기각

# t-검정 결과와 윌콕슨 검정 결과가 같으므로 t-검정 결과를 신뢰할 수 있다.