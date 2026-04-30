import tensorflow as tf
print('version check : ', tf.__version__)
print('즉시 실행 모드 여부 : ',tf.executing_eagerly())
print('GPU 사용 정보 확인 : ', tf.config.list_physical_devices('GPU'))

print('\nTensor : tensorflow에서 data를 담는 기본 자료구조 (숫자 데이터 저장용 다차원 배열)')
# ndarray와 유사하지만 tensorflow에서 연산에 사용되도록 만들어진 객체
print(12, type(12))
print(tf.constant(12))        # 0d tensor (scalar)
print(tf.constant([12]))      # 1d tensor (vector)
print(tf.constant([[12]]))    # 2d tensor (matrix)
print(tf.rank(tf.constant([[12, 1]])))    

tf.print(tf.constant(12))     # tensorflow 전용 출력 함수, 실제값을 중심으로 출력

import numpy as np
# 일반 수치 연산 (CPU 연산이 기본, 자동 미분 불가능, 값 변경 가능)
imsi = np.array([1,2])
print(type(imsi))   # <class 'numpy.ndarray'>

# deeplearning 연산 (CPU와 GPU 연산 가능, 자동 미분 가능, 값 변경 불가능)
a = tf.constant([1,2])      
b = tf.constant([3,4])
print(type(a))      # <class 'tensorflow.python.framework.ops.EagerTensor'>

c = a+b
tf.print(c)

d = tf.constant([3])
e= c + d
tf.print(e)     # broadcast 연산

# numpy, python <-> tensorflow 형변환 가능
print(tf.convert_to_tensor(7))
print(tf.constant(7).numpy())

arr = np.array([1,2])
tfarr = tf.add(arr, 5)      # 텐서 연산시 텐서 타입으로 자동 변환
print(tfarr, type(tfarr))   
print(np.add(tfarr, 2))     # 배열 연산시 넘파이 타입으로 자동 변환

# tf.Variable() : 텐서플로에서 변수 텐서를 만들 때 사용.   예) weight, bias ...
v1 = tf.Variable(1.0)
v2 = tf.Variable(tf.ones(2,))
v3 = tf.Variable(tf.zeros(2,))

v1.assign(123)      # 변수 값 변경, 그냥 할당하면 error
print(v1)

v2.assign([30,40])
print(v2)

aa = tf.Variable(tf.zeros((2,1)))     # 2행 1열에 모두 1을 기억
tf.print('aa : ', aa)

aa.assign(tf.ones((2,1)))
tf.print('aa : ', aa)

aa.assign_add([[2],[3]])
tf.print('aa : ', aa)

a = tf.constant(5)
b = tf. constant(10)

# 조건 설정 cond
result = tf.cond(a < b, lambda:tf.add(10,a), lambda:tf.square(a))
print(result)

# autograph 기능 : 파이썬 코드를 텐서플로 그래프(연산) 코드로 자동 변환
# 텐서플로 2가지 실행 방법
# Eager Execution : 파이썬 코드 처럼 즉시 실행 (기본)
# Graph Execution : 별도 운영이 가능한 계산 그래프를 만들어 최적화 후 실행
# @tf.function 안에서 if, for, while, break, continue, return 등을 사용하면 AutoGraph가 개입함

# 조건 설정 cond를 autoGraph로 
@tf.function             # 이 파이썬 함수를 TensorFlow 계산 그래프로 자동 변환하란 의미
def calcFunc(a,b):
    return tf.add(10,a) if a<b else tf.square(a)

result2 = calcFunc(a,b)
print('result2 : ', result2)

# 반복문 처리
@tf.function
def calcFunc2(n):
    hap = tf.constant(0)
    for i in tf.range(1,n+1):
        hap += i
    return hap
print('hap : ', calcFunc2(10))

imsi = tf.constant(0)
@tf.function
def calcFunc3():
    global imsi     # 함수 밖 변수 사용시 global 필수
    su=1
    for _ in tf.range(3):
        # imsi += su      # 파이썬 연산자 사용
        imsi = tf.add(imsi, su)     # tensor 연산자(권장)

    return imsi
print('imsi : ', calcFunc3())