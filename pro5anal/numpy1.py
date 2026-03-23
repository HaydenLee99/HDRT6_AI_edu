import numpy as np
# NumPy: 내부가 C로 구현되어 있어 연산 속도가 빠름
# 파이썬 list보다 연속된 메모리 구조를 사용하여 효율적인 계산 가능
# 벡터화(vectorization)를 통해 반복문 없이 빠른 배열 연산 지원
# slicing, indexing, broadcasting 등 다양한 기능 제공
# ndarray는 백터/행렬 연산이 가능한 다차원 수치 데이터 구조

ss = ['tom','james', 'oscar', 1, True]
print(ss,' ', type(ss))
print(type(ss[-1]))

ss2 = np.array(ss)
print(ss2,' ', type(ss2))
print(type(ss2[-1]))
# ndarray : 공백으로 구분. numpy.ndarray type.
# 같은 type의 자료로만 구성됨

li = list(range(1, 10))
print(li)
print(li[0], ' ', id(li[0]))
print(li*2)
for i in li:
    print(i * 2, end=" ")
print("\n")

li2 = np.array(li)
print(li2)
print(li2[0], ' ', id(li2[0]))
print(li2*2)
a = np.array([1,2,"3.5"], dtype="float32")
print(a, type(a))       # ndarray는 동일 타입만 취급

b=np.array([[1,2,3],[4,5,6]])
print(b.shape,' ',b[0,0],' ',b[[0]])

c=np.zeros((2,2))
print(c)

d=np.ones((2,2))
print(d)

e=np.eye(3)
print(e)

print()

print(np.random.rand(5))        # 균등 분포(0~1 사이 난수)
print(np.random.randn(5))       # 정규 분포

np.random.seed(0)               # 난수표 0번 값
print(np.random.randn(2,3))

print(list(range(10)))
print(np.arange(10))

# indexing & slicing
a = np.array([1,2,3,4,5])
print(a,' ',a[1])
print(a[1:4])
print(a[1:])
print(a[1:5:2])
print(a[-2:])
b=a
print(a[0],' ',b[0])
b[0] = 88
print(a[0],' ',b[0])
c=np.copy(a)
print(a[0],' ',c[0])
a[0]=1
print(a[0],' ',c[0])