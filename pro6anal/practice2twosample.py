import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sqlalchemy import create_engine


# [two-sample t 검정 : 문제1] 
# 다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 
# 포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.
print("[two-sample t 검정 : 문제1]")
# 수집된 자료 :  
blue = [70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80]
red = [60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66]

# 귀무 : 포장지 색상에 따른 제품의 매출액 차이는 없다.
# 대립 : 포장지 색상에 따른 제품의 매출액 차이는 있다.


if stats.shapiro(blue).pvalue > 0.05 and stats.shapiro(red).pvalue > 0.05:
    print("정규성 OK")

    if stats.levene(blue, red).pvalue > 0.05:
        print("등분산 OK → 일반 t-test")
        answer = stats.ttest_ind(blue, red, equal_var=True)
    else:
        print("등분산 X → Welch t-test")
        answer = stats.ttest_ind(blue, red, equal_var=False)

else:
    print("정규성 X → Mann-Whitney U test")
    answer = stats.mannwhitneyu(blue, red, alternative='two-sided')

if answer.pvalue > 0.05:
    print(f"test result : pvalue는 {answer.pvalue:.4f}로 유의 수준 0.05 보다 크므로 귀무가설 채택 (포장지 색상에 따른 제품의 매출액 차이 없음)")
else:
    print(f"test result : pvalue는 {answer.pvalue:.4f}로 유의 수준 0.05 보다 작으므로 귀무가설 기각 (포장지 색상에 따른 제품의 매출액 차이 있음)")

# 정규성 검정 결과 
# 데이터가 정규성을 만족하고, 등분산 가정이 만족되므로 일반 독립표본 t검정을 수행하였다. 
# 검정 결과 p-value=0.0083으로 유의수준 0.05보다 작으므로 귀무가설을 기각하며, 
# 포장지 색상에 따라 제품 매출액에 통계적으로 유의한 차이가 있음을 확인할 수 있다.


# [two-sample t 검정 : 문제2]  
# 아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
print("\n[two-sample t 검정 : 문제2]")
# 수집된 자료 :  

man = [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
woman = [1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]
# print("남자 자료 건수", len(man))         # 23
# print("여자 자료 건수", len(woman))       # 20

# 귀무 : 성별에 따른 콜레스테롤 양에 차이는 없다.
# 대립 : 성별에 따른 콜레스테롤 양에 차이는 있다.

man_sample = np.random.choice(man, size=15, replace=False)
woman_sample = np.random.choice(woman, size=15, replace=False)

if stats.shapiro(man_sample).pvalue > 0.05 and stats.shapiro(woman_sample).pvalue > 0.05:
    print("정규성 OK")

    if stats.levene(man_sample, woman_sample).pvalue > 0.05:
        print("등분산 OK → 일반 t-test")
        answer = stats.ttest_ind(man_sample, woman_sample, equal_var=True)
    else:
        print("등분산 X → Welch t-test")
        answer = stats.ttest_ind(man_sample, woman_sample, equal_var=False)

else:
    print("정규성 X → Mann-Whitney U test")
    answer = stats.mannwhitneyu(man_sample, woman_sample, alternative='two-sided')

if answer.pvalue > 0.05:
    print(f"test result : pvalue는 {answer.pvalue:.4f}로 유의 수준 0.05 보다 크므로 귀무가설 채택 (성별에 따른 차이 없음)")
else:
    print(f"test result : pvalue는 {answer.pvalue:.4f}로 유의 수준 0.05 보다 작으므로 귀무가설 기각 (성별에 따른 차이 있음)")

# 정규성 검정 결과 
# 두 집단 모두 정규성을 만족하지 않아 비모수 검정인 Mann-Whitney U 검정을 수행하였다. 
# 검정 결과 p-value=0.5895로 유의수준 0.05보다 크므로 귀무가설을 채택하며, 
# 남녀 성별에 따른 콜레스테롤 양의 차이는 통계적으로 유의하지 않다.


# [two-sample t 검정 : 문제3]
# DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
# 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.
print("\n[two-sample t 검정 : 문제3]")

# DB 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123',
    'database': 'test',
    'port': 3306,
    'charset': 'utf8mb4'
}

# ---------------------------
# 1. SQL에서 원본 데이터 가져오기
# ---------------------------
sql = """
SELECT 
    b.busername AS 부서명,
    j.jikwonpay AS 연봉
FROM jikwon j
INNER JOIN buser b ON j.busernum = b.buserno
WHERE b.busername IN ('총무부', '영업부');
"""

engine = create_engine("mysql+pymysql://root:123@localhost:3306/test?charset=utf8mb4")
df = pd.read_sql(sql, engine)

# ---------------------------
# 2. 결측치 처리 (부서별 평균으로 채우기)
# ---------------------------
df['연봉'] = df.groupby('부서명')['연봉'].transform(lambda x: x.fillna(x.mean()))

# ---------------------------
# 3. 부서별 데이터 분리
# ---------------------------
youngup = df[df['부서명'] == '영업부']['연봉'].astype(float)
chongmu = df[df['부서명'] == '총무부']['연봉'].astype(float)

# ---------------------------
# 4. 정규성 검사
# ---------------------------
y_p = stats.shapiro(youngup).pvalue
c_p = stats.shapiro(chongmu).pvalue
print(f"영업부 정규성 p-value: {y_p:.4f}")
print(f"총무부 정규성 p-value: {c_p:.4f}")

# ---------------------------
# 5. 검정 선택
# ---------------------------
if y_p > 0.05 and c_p > 0.05:
    levene_p = stats.levene(youngup, chongmu).pvalue
    print(f"등분산 검정 p-value: {levene_p:.4f}")

    if levene_p > 0.05:
        print("→ 일반 t-test")
        result = stats.ttest_ind(youngup, chongmu, equal_var=True)
    else:
        print("→ Welch t-test")
        result = stats.ttest_ind(youngup, chongmu, equal_var=False)
else:
    print("→ Mann-Whitney U test")
    result = stats.mannwhitneyu(youngup, chongmu, alternative='two-sided')

# ---------------------------
# 6. 결과 해석
# ---------------------------
print(f"\n검정 통계량: {result.statistic:.2f}")
print(f"p-value: {result.pvalue:.4f}")

if result.pvalue > 0.05:
    print("귀무가설 채택: 부서별 연봉 차이 없음")
else:
    print("귀무가설 기각: 부서별 연봉 차이 있음")

# 정규성 검정 결과 
# 두 집단 모두 정규성을 만족하지 않아 비모수 검정인 Mann-Whitney U 검정을 수행하였다. 
# 검정 결과 p-value=0.4721로 유의수준 0.05보다 크므로, 부서별 연봉 차이는 통계적으로 유의하지 않다.