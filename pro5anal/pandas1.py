import pandas as pd
from pandas import Series, DataFrame
import numpy as np
# pandas : 데이터 분석과 조작을 위한 Python 라이브러리.
# DataFrame과 Series 자료구조를 제공하여, 표 형식 데이터 처리 용이.

# Series : 일련의 객체를 담을 수 있는 1차원 배열과 같은 자료구조로 색인(index)을 갖는다.
# 요소 값은 object type
obj = pd.Series([3, 7, -5, '사'])
# obj = pd.Series({3, 7, -5, '4'})       # TypeError: set은 순서가 없다.
print(obj, type(obj))

obj2 = pd.Series([3, 7, -5, 4], index=['a','b','c','d'])
print(obj2)
print(obj2.sum(),' ',np.sum(obj2),' ',sum(obj2))
print(obj2.std())
print(obj2.values)
print(obj2.index)

print(obj2['a'])        # 3
print(obj2[['a']])      # a     3

print(obj2[['a','b']])
print(obj2['a':'c'])

print(obj2[2])
print(obj2.iloc[2])
print(obj2[1:4])

print(obj2[[2,1]])
print(obj2.iloc[[2,1]])

print('a' in obj2)
print('q' in obj2)

print("python dict 자료를 Series 객체로 생성")
names={'a':5000,'b':2500,'c':1250}
obj3 = Series(names)
print(names)
print(obj3)
obj3.index = ['오천','이천오백','천이백오십']
print(obj3)
obj3.name = '숫자 이름'
print(obj3)

print("DataFrame 객체----------------------")
df = pd.DataFrame(obj3)

data = {
    'name':['john','f','k'],
    'addr':('seoul','japan',"usa"),
    'age':[41,87,64]
}
frame = DataFrame(data)
print(frame,'\n')

print(frame['name'])
print(frame.name)

print(DataFrame(data=data, columns=['name','age','addr']))
# NaN(결측치)
frame2 = DataFrame(data,columns=['name','age','addr','tel'], index=['a','b','c'])
frame2['tel']='111-1111'
print(frame2)

val = Series(['222-2222','333-3333'], index=['b','c'])
print(val)
frame2['tel'] = val
print(frame2)

print(frame2.T)
print(frame2.values)        # 결과는 list 타입
print(frame2.values[0,1])

frame3=frame2.drop('b', axis=0)
print(frame3)

frame4=frame2.drop('tel', axis=1)
print(frame4)

print(frame2.sort_index(axis=0, ascending=False))
print(frame2.sort_index(axis=1, ascending=True))

print(frame2.rank(axis=0))
counts = frame2['addr'].value_counts()
print(counts)

# 문자열 자르기
data = {
    'juso':['강남구 역삼동','중구 신당동', '강남구 대치동'],
    'inwon':[23,25,15]
}
fr = DataFrame(data)
print(fr)
result1 = Series([x.split()[0] for x in fr.juso])
result2 = Series([x.split()[1] for x in fr.juso])
print(result1)
print(result2)
print(result1.value_counts())