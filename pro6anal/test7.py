# 집단간 차이분석 : 평균 or  비율 차이를 분석
# 모집단에서 추출한 표본 정보를 이용하여 모집단의 다양한 특성을 과학적으로 추론 가능

# 단일 표본 t 검정
# 정규 분포의 표본에 대한 기대값을 조사하는 검정 방법.
# 예상 평균값과 표본 자료간에 평균 차이를 검정
# 독립변수:범주형 , 종속변수:연속형
# 하나의 집단에 대한 표본 평균이 예측된 평균(모집단)과 같은가 확인

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

# 실습1 : 어느 남성 집단의 평균 키 검정
# 귀무  : 해당 집단의 평균 키가 <177> 이다. (모수)
# 대립  : 해당 집단의 평균 키가 <177> 이 아니다.
one_sample = [167.0, 182.7, 169.6, 176.8, 185.0]
print(np.array(one_sample).mean())      # 176.22
result = stats.ttest_1samp(one_sample, popmean=177)     # (데이터, 모수의 평균)
print(result)
# statistic=-0.2213944, pvalue=0.83562821, df=4
# 해석 : 유의수준 0.05 < pvalue 0.83562821 귀무가설 채택

result2 = stats.ttest_1samp(one_sample, popmean=165)
print(result2)
# statistic=3.184, pvalue=0.033, df=4
# 해석 : 유의수준 0.05 > pvalue 0.033 귀무가설 기각

sns.displot(one_sample, bins=10, kde=True)
plt.xlabel('data')
plt.ylabel('value')
plt.show()

