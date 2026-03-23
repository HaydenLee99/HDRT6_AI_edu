# JSON 자료 : XML에 비해 경량. 배열 개념만 있으면 처리 가능
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

dict = {'name':'tom', 'age':25, 'score':['85','88','91']}
print(type(dict))

print('JSON 인코딩 : dict를 str로')

str_val = json.dumps(dict, indent=4)        # indent 들여쓰기 하여 출력
print(str_val, type(str_val))

print(str_val[0:5])

print('JSON 디코딩 : str을 dict로 ')
json_val = json.loads(str_val)
print(json_val, type(json_val))
print(json_val['name'])
for k in json_val.keys():
    print(k)
for v in json_val.values():
    print(v)


print("\n서울시 제공 도서관 정보 json 읽기")
url = "http://openapi.seoul.go.kr:8088/sample/json/SeoulLibraryTimeInfo/1/5/"

import urllib.request as req
import json
import pandas as pd

# 1. 요청 및 JSON 파싱
plainText = req.urlopen(url).read().decode()
jsonData = json.loads(plainText)

# 2. 첫 번째 도서관 이름 출력
print(jsonData["SeoulLibraryTimeInfo"]["row"][0]["LBRRY_NAME"])
print()

# 3. 전체 데이터 수집
libData = jsonData["SeoulLibraryTimeInfo"]["row"]
data = []
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    addr = ele.get('ADRES')
    data.append([name, tel, addr])

# 4. DataFrame 생성
df = pd.DataFrame(data, columns=['도서관명', '전화', '주소'])
print(df)