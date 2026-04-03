# 정렬 알고리즘
# list 안에 자료를 오름차순으로 정렬

# 2. 삽입 정렬 : 앞에서 부터 순서대로 꺼내서 적합한 자리에 끼어넣는 정렬
#   방법1: 이해 위주
def find_ind_idx(r, v):
    for i in range(len(r)):
        if v < r[i]:
            return i
        
    # 적합한 위치를 찾지 못하였을 경우, 맨 뒤에 삽입한다.
    return len(r)

d = [2,4,5,1,3]
# print(find_ind_idx(d, 1))
def ins_sort(a):
    result = []
    while a:
        value = a.pop(0)
        ins_idx = find_ind_idx(result, value)
        result.insert(ins_idx, value)
    return result
print(ins_sort(d))

#   방법2: 일반 알고리즘
def ins_sort2(a):
    n = len(a)
    # 두 번째 부터 마지막 까지 차례대로 삽입할 대상을 선택
    for i in range(1,n):
        key = a[i]    # i번 위치에 있는 값을 key에 저장
        j = i-1       # j를 i 바로 왼쪽 위치로
        while j >= 0 and a[j] > key:
            a[j+1] = a[j]       # 삽입할 공간이 생기도록 값을 우측으로 밀기
            j -= 1              # 그 다음 왼쪽으로 이동하면서 다시 비교 작업 수행
        a[j+1] = key  # 찾은 위치에 key를 저장
d = [2,4,5,1,3]
ins_sort2(d)
print(d)