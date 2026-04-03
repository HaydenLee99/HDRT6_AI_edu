# 정렬 알고리즘
# list 안에 자료를 오름차순으로 정렬

# 5.버블 정렬 : 이웃한 요소를 비교해가며 정렬
# 재귀함수 사용

#   방법1: 이해 위주
def bubble_sort(a):
    n=len(a)
    while True:
        swap_flag = False
    
        for i in range(0, n-1):
            if a[i] > a[i+1]:
                print(a)
                a[i], a[i+1] = a[i+1], a[i]
                swap_flag = True
                
        if swap_flag ==  False:
            return

d = [5, 2, 8, 1, 7, 3, 4, 6]
bubble_sort(d)
print(d)