# array 연산
import numpy as np

x=np.array([[1,2],[3,4]], dtype=np.float32)
print(x,' ', x.dtype)

y=np.arange(5,9).reshape(2,2).astype(np.float32)
print(y,' ', y.dtype)

print("1. 배열 더하기")
print(x+y)                   # 파이썬 연산자 (상대적으로 느림)
print(np.add(x,y))           # numpy 함수
print()

print("2. 배열 빼기")
print(x-y)                   # 파이썬 연산자 (상대적으로 느림)
print(np.subtract(x,y))      # numpy 함수
print()

print("3. 배열 곱하기")
print(x*y)                   # 파이썬 연산자 (상대적으로 느림)
print(np.multiply(x,y))      # numpy 함수
print()

print("4. 배열 나누기")
print(x/y)                   # 파이썬 연산자 (상대적으로 느림)
print(np.divide(x,y))        # numpy 함수
print()

print("5. 벡터 행렬곱")
v=np.array([9,10]); w=np.array([11,12])
print(v.dot(w))             # 파이썬 연산자 (상대적으로 느림)
print(np.dot(v,w))          # numpy 함수
print()

print(x)
print(np.max(x),' ',np.min(x))
print(np.argmax(x),' ',np.argmin(x))       # 값이 있는 인덱스가 어딘지 출력
print(np.cumsum(x))                     # 누적합
print(np.cumprod(x))                    # 누적곱

names1 = np.array(['tom','james','oscar', 'tom'])
names2 = np.array(['tom','smith','hayden'])

print(np.unique(names1))

print(np.intersect1d(names1,names2))        # 교집합
print(np.intersect1d(names1,names2, assume_unique=True))        # 교집합(중복허용)

print(np.union1d(names1,names2))        # 합집합

# 전치
print(x.T)
print(x.transpose())
print(x.swapaxes(0, 1))

print('broadcasting : 크기가 다른 배열 간의 연산 - 작은 배열을 여러번 반복해 큰 배열과 연산')
x=np.arange(1,10).reshape(3,3)      # 3X3
y=np.array([1,0,1])                 # 1X3
print(x)
print(y)
print(x+y)

# 배열 file i/o
# np.savetxt("my.txt", x)
# np.loadtxt("my.txt")

