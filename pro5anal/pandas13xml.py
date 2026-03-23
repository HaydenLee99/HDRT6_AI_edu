# bs4로 XML 모듈 처리
from bs4 import BeautifulSoup

with open('my.xml', mode='r', encoding='utf-8') as f:
    xmlfile = f.read()
    print(xmlfile, type(xmlfile))

soup = (BeautifulSoup(xmlfile, 'lxml'))
print(type(soup))
itemTag = soup.find_all('item')
print(itemTag[1])

print()
nameTag = soup.find_all('name')
print(nameTag[0]['id'])

print('-'*20)
for i in itemTag:
    nameTag = i.find_all('name')
    for j in nameTag:
        print("id: " + j["id"] + "\tname:" + j.string)
        tel = i.find("tel")
        print("tel:", tel.string)
    
    for j in i.find_all("exam"):
        print("kor:" + j["kor"] + " eng:" + j["eng"])
        print()


print("\n서울시 제공 도서관 정보 XML 읽기")
import urllib.request as req
import pandas as pd

url = "http://openapi.seoul.go.kr:8088/sample/xml/SeoulLibraryTimeInfo/1/5/"
plainText = req.urlopen(url).read().decode()
# print(plainText)
xmlObj = BeautifulSoup(plainText, 'xml')
libData = xmlObj.select("row")
# print(libData)
rows = []
for data in libData:
    name = data.find("LBRRY_NAME").text.strip()
    addr = data.find("ADRES").text.strip()
    # print(name, ':',addr)
    rows.append({"도서관명":name, "주소":addr})
df = pd.DataFrame(rows)
print(df)
print("건수 : ", len(df))