# css 셀럭터 이용 연습
from bs4 import BeautifulSoup

html_page="""
<html>
<body>
    <div id = "hello">
        <a href="https://www.naver.com">naver</a><br>
        <span>
            <a href="https://www.daum.net">daum</a><br>
        </span>
        <ul class="world">
            <li>안녕</li>
            <li>반가워</li>
        </ul>
    </div>
    <div id="hi" class="good">
        <b>두번째 div</b>
    </div>
</body>
</html>
"""
soup = BeautifulSoup(html_page, 'lxml')

# aa = soup.select_one("div")
# aa = soup.select_one("div#hello")           # id가 hello인 div 하나만
# aa = soup.select_one("div.good")            # class가 good인 div 하나만
aa = soup.select_one("div#hello > a")       # div에 id가 hello에 직계자식 a 하나만
# aa = soup.select_one("div#hello a")       # div에 id가 hello에 직계자손 a 하나만
print("aa : ", aa, ' ', aa.string)          # aa.string : div 안에 하위값이 있으므로 None 반환

print()
# bb = soup.select("div")
# bb = soup.select("div#hello > ul.world")
# bb = soup.select("div#hello ul.world")
bb = soup.select("div#hello ul.world > li")
print("bb : ", bb)
for i in bb:
    print(i)
    print('-'*10)
    print(i.text)
    print()

print("-----위키백과 사이트에서 이순신으로 검색된 자료 읽기-----")
import requests

url = "https://ko.wikipedia.org/wiki/이순신"
headers = {"User-Agent":"Mozilla/5.0"}

wiki = requests.get(url=url, headers=headers)
# print(wiki.text[:100])

soup = BeautifulSoup(wiki.text, 'lxml')
# result = soup.select("p#mwHw")                # 한 문단 읽기
result = soup.select("#mw-content-text p")      # 전문 읽기
# print(result)
for s in result:
    for sup in s.find_all("sup"):
        sup.decompose()             # decompose : tag 삭제
    print(s.get_text(strip=True))

print("--- 교촌 치킨 사이트에서 메뉴, 가격 자료 읽고 활용 연습 ---")
from pandas import DataFrame
import re

url = "https://kyochon.com/menu/chicken.asp"
headers = {"User-Agent":"Mozilla/5.0"}

response = requests.get(url=url, headers=headers)
# print(response.text)
soup2 = BeautifulSoup(response.text, 'lxml')

# 메뉴명 얻기
# names = soup2.select("dl.txt > dt")
# print(names)
names = [tag.text.strip() for tag in soup2.select("dl.txt > dt")]
# print(names)

# 가격 얻기
prices = [int(tag.text.strip().replace(',', '')) for tag in soup2.select("p.money strong")]
# print(prices)

df = DataFrame({"상품명":names, "가격":prices})
print(df.head(3),'\n')
print(f"교쵼 가격 평균 : {df['가격'].mean():.2f} 원")
print(f"교쵼 가격 표준편차 : {df['가격'].std():.2f} 원")
print(f"교쵼 가격 변동계수(CV) : {df['가격'].std()/df['가격'].mean()*100:.2f} %")
# 해석 : CV 28.31% 이므로 평균 대비 중간 수준으로 퍼져 있다.