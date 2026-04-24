# 선형구조 연습

""" 놀이공원 줄서기 """
# 연습 1 - python 함수 사용
line = ['철수', '영희', '민수']
print('현재 줄 상태 : ', line)

# 명수 새치기
line.insert(1,'명수')
print('현재 줄 상태 : ', line)

# 영희 이탈
line.remove('영희')
print('현재 줄 상태 : ', line)

# 앞사람 부터 놀이기구 탑승
line.pop(0)
print('현재 줄 상태 : ', line)
line.pop(0)
print('현재 줄 상태 : ', line)
line.pop(0)
print('현재 줄 상태 : ', line)