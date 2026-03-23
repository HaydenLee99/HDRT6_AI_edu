import numpy as np
from pandas import Series, DataFrame, read_csv, cut, unique
import os

# pandas 문제 1)
#   a) 표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오. 
df1 = DataFrame(np.random.randn(9, 4))
print(df1)
#   b) a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오
df1.columns = 'No1', 'No2', 'No3', 'No4'
print(df1)
#   c) 각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용
print(df1.mean(axis=1))

# pandas 문제 2)
# a) DataFrame으로 위와 같은 자료를 만드시오. colume(열) name은 numbers, row(행) name은 a~d이고 값은 10~40.
df2 = DataFrame(np.arange(10,41,10).reshape(4,1), columns=['numbers'], index=list('abcd'))
print(df2)
# b) c row의 값을 가져오시오.
print(df2.loc['c',])
# c) a, d row들의 값을 가져오시오.
print(df2.loc[['a','d']])
# d) numbers의 합을 구하시오.
print(df2['numbers'].sum())
# e) numbers의 값들을 각각 제곱하시오.
print(df2['numbers'].mul(df2['numbers']))
# f) floats 라는 이름의 칼럼을 추가하시오. 값은 1.5, 2.5, 3.5, 4.5
df2['floats'] = 1.5, 2.5, 3.5, 4.5
print(df2)
# g) names라는 이름의 칼럼을 위의 결과에 추가하시오. Series 클래스 사용.
# d 길동
# a 오정
# b 팔계
# c 오공
names = Series(['길동','오정','팔계','오공'], index=['d','a','b','c'])
df2['names'] = names
print(df2)

# pandas 문제 3)
# 1) 5 x 3 형태의 랜덤 정수형 DataFrame을 생성하시오. (범위: 1 이상 20 이하, 난수)
df3 = DataFrame(np.random.randint(1,21,15).reshape(5,3))
print(df3)
# 2) 생성된 DataFrame의 컬럼 이름을 A, B, C로 설정하고, 행 인덱스를 r1, r2, r3, r4, r5로 설정하시오.
df3.columns=['A','B','C']
df3.index=['r1','r2','r3','r4','r5']
print(df3)
# 3) A 컬럼의 값이 10보다 큰 행만 출력하시오.
print(df3[df3['A'] > 10])
# 4) 새로 D라는 컬럼을 추가하여, A와 B의 합을 저장하시오.
df3['D'] = df3['A'].add(df3['B'])
print(df3)
# 5) 행 인덱스가 r3인 행을 제거하되, 원본 DataFrame이 실제로 바뀌도록 하시오.
df3 = df3.drop('r3')
print(df3)
# 6) 아래와 같은 정보를 가진 새로운 행(r6)을 DataFrame 끝에 추가하시오.
#       A   B     C     D
# r6   15   10    2   (A+B)
df3.loc['r6'] = [15, 10, 2, 15+10]
print(df3)

# pandas 문제 4)
# 다음과 같은 재고 정보를 가지고 있는 딕셔너리 data가 있다고 하자.
data = {
    'product': ['Mouse', 'Keyboard', 'Monitor', 'Laptop'],
    'price':   [12000,     25000,      150000,    900000],
    'stock':   [  10,         5,          2,          3 ]
}
# 1) 위 딕셔너리로부터 DataFrame을 생성하시오. 단, 행 인덱스는 p1, p2, p3, p4가 되도록 하시오.
df4 = DataFrame(data,index=['p1','p2','p3','p4'])
print(df4)
# 2) price와 stock을 이용하여 'total'이라는 새로운 컬럼을 추가하고, 값은 'price x stock'이 되도록 하시오.
df4['total'] = df4['price'].mul(df4['stock'])
print(df4)
# 3) 컬럼 이름을 다음과 같이 변경하시오. 원본 갱신
#    product → 상품명,  price → 가격,  stock → 재고,  total → 총가격
df4.columns = ['상품명', '가격', '재고', '총가격']
print(df4)
# 4) 재고(재고 컬럼)가 3 이하인 행의 정보를 추출하시오.
print(df4[df4['재고'] <= 3])
# 5) 인덱스가 p2인 행을 추출하는 두 가지 방법(loc, iloc)을 코드로 작성하시오.
print(df4.loc['p2'])
print(df4.iloc[1])
# 6) 인덱스가 p3인 행을 삭제한 뒤, 그 결과를 확인하시오. (원본이 실제로 바뀌지 않도록, 즉 drop()의 기본 동작으로)
print(df4.drop('p3', axis=0))
# 7) 위 DataFrame에 아래와 같은 행(p5)을 추가하시오.
#             상품명             가격     재고    총가격
#  p5       USB메모리    15000     10    가격*재고
df4.drop('p3', axis=0)
df4.loc['p5'] = ['USB메모리',15000,10,15000*10]
print(df4)

# pandas 문제 5)  타이타닉 승객 데이터를 사용하여 아래의 물음에 답하시오.
# 데이터 : https://github.com/pykwon/python/blob/master/testdata_utf8/titanic_data.csv
# raw_url : https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv
# df5 = read_csv('data/titanic_data.csv')
raw_url = r"https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv"
df5 = read_csv(raw_url, encoding='utf8')
print(df5.info())

#  열 구성 정보
#    Survived : 0 = 사망, 1 = 생존
#    Pclass : 1 = 1등석, 2 = 2등석, 3 = 3등석
#    Sex : male = 남성, female = 여성
#    Age : 나이
#    SibSp : 타이타닉 호에 동승한 자매 / 배우자의 수
#    Parch : 타이타닉 호에 동승한 부모 / 자식의 수
#    Ticket : 티켓 번호
#    Fare : 승객 요금
#    Cabin : 방 호수
#    Embarked : 탑승지, C = 셰르부르, Q = 퀸즈타운, S = 사우샘프턴

#  1) 데이터프레임의 자료로 나이대(소년, 청년, 장년, 노년)에 대한 생존자수를 계산한다.
#  cut() 함수 사용
bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]


#   2) 성별 및 선실에 대한 자료를 이용해서 생존여부(Survived)에 대한 생존율을 피봇테이블 형태로 작성한다. 
#  df.pivot_table()
#  index에는 성별(Sex)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.
#  index에는 성별(Sex) 및 나이(Age)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.





# ---------------------------------------------------


# pandas 문제 6)
#  https://github.com/pykwon/python/tree/master/testdata_utf8
#  1) human.csv 파일을 읽어 아래와 같이 처리하시오.
raw_url = r"https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/human.csv"
df6_human = read_csv(raw_url, encoding='utf8')
col = []
for i in df6_human.columns:
    a = i.strip()
    col.append(a)
df6_human.columns = col
#      - Group이 NA인 행은 삭제

#      - Career, Score 칼럼을 추출하여 데이터프레임을 작성
#      - Career, Score 칼럼의 평균계산

#  참고 : strip() 함수를 사용하면 주어진 문자열에서 양쪽 끝에 있는 공백과 \n 기호를 삭제시켜 준다. 
#        strip() 함수는 문자열의 양 끝에 있는 공백과 \n을 제거. (중간 공백 제거 하지 않음)

os._exit(0)
#  2) tips.csv 파일을 읽어 아래와 같이 처리하시오.
raw_url = r"https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tips.csv"
df6_tips = read_csv(raw_url, encoding='utf8')

# - 파일 정보 확인
print(df6_tips)

# - 앞에서 3개의 행만 출력
print(df6_tips.head(3))

# - 요약 통계량 보기
print(df6_tips.describe())

# - 흡연자, 비흡연자 수를 계산  : value_counts()
print("비흡연자 : ", len(df6_tips[df6_tips['smoker']=='No']))
print("흡연자 : ", len(df6_tips[df6_tips['smoker']=='Yes']))

# - 요일을 가진 칼럼의 유일한 값 출력  : unique()
print(unique(df6_tips['day']))


# pandas 문제 7)

#  a) MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.
#      - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
#      - DataFrame의 자료를 파일로 저장
#      - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
#      - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
#      - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
#      - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성


#  b) MariaDB에 저장된 jikwon 테이블을 이용하여 아래의 문제에 답하시오.
#      - pivot_table을 사용하여 성별 연봉의 평균을 출력
#      - 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프
#      - 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))



#  c) 키보드로 사번, 직원명을 입력받아 로그인에 성공하면 console에 아래와 같이 출력하시오.
#       조건 :  try ~ except MySQLdb.OperationalError as e:      사용
#      사번  직원명  부서명   직급  부서전화  성별
#      ...
#      인원수 : * 명





# pandas 문제 8)
# MariaDB에 저장된 jikwon, buser 테이블을 이용하여 아래의 문제에 답하시오.
# Django(Flask) 모듈을 사용하여 결과를 클라이언트 브라우저로 출력하시오.

#    1) 사번, 직원명, 부서명, 직급, 연봉, 근무년수를 DataFrame에 기억 후 출력하시오. (join)
#        : 부서번호, 직원명 순으로 오름 차순 정렬 
#    2) 부서명, 직급 자료를 이용하여  각각 연봉합, 연봉평균을 구하시오.
#    3) 부서명별 연봉합, 평균을 이용하여 세로막대 그래프를 출력하시오.
#    4) 성별, 직급별 빈도표를 출력하시오.










# -----------------
os._exit(0)
"""
pandas 문제 5)  타이타닉 승객 데이터를 사용하여 아래의 물음에 답하시오.

타이타닉호 3D 이미지-마젤란사(심해지도 제작 업체) 홈페이지
  데이터 : https://github.com/pykwon/python/blob/master/testdata_utf8/titanic_data.csv
   titanic_data.csv 파일을 다운로드 후
   df = pd.read_csv('파일명',  header=None,,,)  

  열 구성 정보
    Survived : 0 = 사망, 1 = 생존
    Pclass : 1 = 1등석, 2 = 2등석, 3 = 3등석
    Sex : male = 남성, female = 여성
    Age : 나이
    SibSp : 타이타닉 호에 동승한 자매 / 배우자의 수
    Parch : 타이타닉 호에 동승한 부모 / 자식의 수
    Ticket : 티켓 번호
    Fare : 승객 요금
    Cabin : 방 호수
    Embarked : 탑승지, C = 셰르부르, Q = 퀸즈타운, S = 사우샘프턴

 1) 데이터프레임의 자료로 나이대(소년, 청년, 장년, 노년)에 대한 생존자수를 계산한다.
      cut() 함수 사용
     bins = [1, 20, 35, 60, 150]
      labels = ["소년", "청년", "장년", "노년"]
"""

import pandas as pd
import numpy as np

df=pd.read_csv('titanic.csv')

bins=[1,20,35,60,150]
labels=["소년","청년","장년","노년"]
df['나이대']=pd.cut(df['Age'],bins=bins,labels=labels)
result=df.groupby('나이대',observed=True)['Survived'].sum()
result=result.reset_index()
result.columns=['나이대','생존자수']
print(result)
print()



"""
  2) 성별 및 선실에 대한 자료를 이용해서 생존여부(Survived)에 대한 생존율을 피봇테이블 형태로 작성한다. 
      df.pivot_table()
     index에는 성별(Sex)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.
      출력 결과 샘플1 :       

pclass	1	2	3
sex			
female	0.968085	0.921053	0.500000
male	0.368852	0.157407	0.135447

   index에는 성별(Sex) 및 나이(Age)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.
   출력 결과 샘플2 : 위 결과물에 Age를 추가. 백분율로 표시. 소수 둘째자리까지.    예: 92.86
"""

import pandas as pd
import numpy as np

df = pd.read_csv('titanic.csv')

# 나이대 컬럼 생성
bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]
df['나이대'] = pd.cut(df['Age'], bins=bins, labels=labels)

# 샘플1 
pivot1 = df.pivot_table(
    values='Survived',
    index='Sex',
    columns='Pclass',
    aggfunc='mean'
)
print(pivot1)
print()

# 샘플2 
pivot2 = df.pivot_table(
    values='Survived',
    index=['Sex', '나이대'],
    columns='Pclass',
    aggfunc='mean'
)
pivot2 = (pivot2 * 100).round(2)
print(pivot2)



"""
pandas 문제 6)

 https://github.com/pykwon/python/tree/master/testdata_utf8

 1) human.csv 파일을 읽어 아래와 같이 처리하시오.
     - Group이 NA인 행은 삭제
     - Career, Score 칼럼을 추출하여 데이터프레임을 작성
     - Career, Score 칼럼의 평균계산

     참고 : strip() 함수를 사용하면 주어진 문자열에서 양쪽 끝에 있는 공백과 \n 기호를 삭제시켜 준다. 
             그래서 위의 문자열에서 \n과 오른쪽에 있는 공백이 모두 사라진 것을 확인할 수 있다. 
             주의할 점은 strip() 함수는 문자열의 양 끝에 있는 공백과 \n을 제거해주는 것이지 중간에 
             있는 것까지 제거해주지 않는다.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/human.csv",skipinitialspace=True)
print(df)
df.columns = df.columns.str.strip()
print(df.dropna(subset=["Group"]))
df1 = df.dropna(subset=["Group"])
print(df1[['Career', 'Score']])
print(df1[['Career', 'Score']].mean())


"""
 2) tips.csv 파일을 읽어 아래와 같이 처리하시오.
     - 파일 정보 확인
     - 앞에서 3개의 행만 출력
     - 요약 통계량 보기
     - 흡연자, 비흡연자 수를 계산  : value_counts()
     - 요일을 가진 칼럼의 유일한 값 출력  : unique()
          결과 : ['Sun' 'Sat' 'Thur' 'Fri']
 """

df3 = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tips.csv")
print(df3.info())
print(df3.head(3))
print(df3.describe())
print(df3["smoker"].value_counts())
print(df3["day"].unique())
