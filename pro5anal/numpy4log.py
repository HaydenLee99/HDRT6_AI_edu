# 편차가 큰 데이터 처리 방법

# (1) 로그 변환 (log transformation)
# - 큰 값과 작은 값의 스케일 차이를 줄여줌
# - 치우친(skewed) 데이터를 정규분포에 가깝게 변환 가능
# - 지수적 관계(y = a^x)를 선형 관계로 바꾸어 모델링 용이
# - 예: y = log(x)

# (2) Min-Max Scaling
# - 데이터 범위를 0 ~ 1 또는 -1 ~ 1 사이로 변환
# - 스케일이 다른 특징(feature) 간 비교를 용이하게 함
# - 예: X_scaled = (X - X_min) / (X_max - X_min)

# (3) 표준화 (Standardization)
# - 평균이 0, 표준편차가 1이 되도록 변환
# - 이상치(outlier)에 상대적으로 덜 민감
# - 예: X_standard = (X - mean) / std

# (4) 정규화 (Normalization)
# - 각 샘플의 벡터 크기를 1로 맞춤 -> 모든 데이터를 0~1 사이 범위 내에서 표시
# - 주로 거리 기반 알고리즘(KNN, SVM)에서 사용
# - 예: X_normalized = X / ||X|| (L2 norm)

import numpy as np
np.set_printoptions(suppress=True, precision=4)

def test():
    values = np.array([345,34.5,3.45,0.345,0.01,0.1,10,100])
    print(np.log2(3.45),' ',np.log10(3.45),' ',np.log(3.4))
    print("원본값\t:\t",values)

    log_val = np.log10(values)
    print('log10\t:\t',log_val)

    ln_val = np.log(values)
    print('ln\t:\t',ln_val)

    # 정규화
    min_log = np.min(log_val)
    max_log = np.max(log_val)
    normalized = (log_val - min_log) / (max_log - min_log)
    print('정규화_결과\t:\t', normalized)

class LogTrans:
    # 편차가 큰 데이터를 로그 스케일로 변환하고 그 역변환을 제공하는 클래스
    def __init__(self, offset:float=1.0):
        self.offset = offset
    # 로그 변환
    def transform(self, x:np.ndarray) -> np.ndarray:
        return np.log(x + self.offset)
    # 로그 역변환
    def inverse_transform(self, x_log:np.ndarray) -> np.ndarray:
        return np.exp(x_log) - self.offset 
    

def main():
    test()
    print('***' * 10)
    data=np.array([0.001,0.01,0.1,10,100,1000,10000], dtype=float)
    
    log_trans = LogTrans(offset=1.0)
    data_log_scaled = log_trans.transform(data)
    reversed_data = log_trans.inverse_transform(data_log_scaled)

    print('원본\t\t:\t',data)
    print('ln 변환\t\t:\t',data_log_scaled)
    print('ln 역변환\t:\t',reversed_data)
    
if __name__=="__main__":
    main()