import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

# [one-sample t 검정 : 문제1]  
# 영사기(프로젝터)에 사용되는 구형 백열전구의 <수명은 250 시간> 이라고 알려졌다. 
# 한국 연구소에서 <수명이 50 시간 더 긴> 새로운 백열전구를 개발하였다고 발표하였다. 
# 연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명 시간 관련 자료를 얻었다. 
# 한국 연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.
# 수집된 자료 :  305 280 296 313 287 240 259 266 318 280 325 295 315 278

Q1_data = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
if len(Q1_data) >= 30:
    result = stats.ttest_1samp(Q1_data, popmean=250+50)
else:
    # 샤피로 정규성 검증
    if stats.shapiro(Q1_data)[1] > 0.05:
        # 정규성을 가지므로 t-검정 시행
        result = stats.ttest_1samp(Q1_data, popmean=250+50)
        print("Q1 one-sample t-test result...")
        if result[1] >= 0.05:
            print(f'유의 수준 0.05 보다 pvalue={result[1]:.6f}가 크므로 귀무가설 채택.\n한국 연구소의 발표 결과를 신뢰할 수 있다.')
        else:
            print(f'유의 수준 0.05 보다 pvalue={result[1]:.6f}가 작으므로 귀무가설 기각.\n한국 연구소의 발표 결과를 신뢰할 수 없다.')
    else:
        # 정규성 위배이므로 wilcoxon 시행
        result = stats.wilcoxon(np.array(Q1_data) - (250+50))
        print("Q1 wilcoxon test result...")
        if result[1] >= 0.05:
            print(f'유의 수준 0.05 보다 pvalue={result[1]:.6f}가 크므로 귀무가설 채택.\n한국 연구소의 발표 결과를 신뢰할 수 있다.')
        else:
            print(f'유의 수준 0.05 보다 pvalue={result[1]:.6f}가 작으므로 귀무가설 기각.\n한국 연구소의 발표 결과를 신뢰할 수 없다.')
print()
# [one-sample t 검정 : 문제2] 
# 국내에서 생산된 대다수의 노트북 <평균 사용 시간이 5.2 시간> 으로 파악되었다. 
# A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 A회사 노트북 <150대를 랜덤하게 선정> 하여 검정을 실시한다.  
# 실습 파일 : one_sample.csv
# 참고 : time에 공백을 제거할 땐 ***.time.replace("     ", ""), null인 관찰값은 제거.
Q2_data_raw = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv")
Q2_data = Q2_data_raw[['time']]
Q2_data = Q2_data.replace("     ", np.nan).dropna().astype(float)

if len(Q2_data) >= 30:
    # 중심극한정리에 의해 정규성을 가지므로 t-검정 시행
    result = stats.ttest_1samp(Q2_data['time'], popmean=5.2)
    print("Q2 one-sample t-test result...")
    if result[1] >= 0.05:
        print(f'유의 수준 0.05 보다 pvalue={result[1]:.6f}가 크므로 귀무가설 채택.\nA회사 노트북 평균 사용 시간을 신뢰할 수 있다.')
    else:
        print(f'유의 수준 0.05 보다 pvalue={result[1]:.6f}가 작으므로 귀무가설 기각.\nA회사 노트북 평균 사용 시간을 신뢰할 수 없다.')
print()
# [one-sample t 검정 : 문제3] 
# https://www.price.go.kr/tprice/portal/main/main.do 에서 
# 메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료(엑셀)를 파일로 받아 미용 요금을 얻도록 하자. 
# 정부에서는 전국 평균 미용 요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오. (월별)
path = r"..\pro5anal\data\개인서비스지역별_동향[2026-02월]331-0시47분.xls"
Q3_data_raw = pd.read_excel(path)
Q3_data = Q3_data_raw[Q3_data_raw.columns[3:]].drop('세종', axis=1).iloc[0]

if len(Q3_data) < 30:
    # 샤피로 정규성 검증
    if stats.shapiro(Q3_data)[1] > 0.05:
        # 정규성을 가지므로 t-검정 시행
        result = stats.ttest_1samp(Q3_data, popmean=15000)
        print("Q3 one-sample t-test result...")
        if result[1] >= 0.05:
            print(f'유의 수준 0.05 보다 pvalue={result[1]:.6f}가 크므로 귀무가설 채택.\n정부 발표를 신뢰할 수 있다.')
        else:
            print(f'유의 수준 0.05 보다 pvalue={result[1]:.6f}가 작으므로 귀무가설 기각.\n정부 발표를 신뢰할 수 없다.')
