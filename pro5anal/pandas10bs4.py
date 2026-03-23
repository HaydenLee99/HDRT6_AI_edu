# BeautifulSoup 객체 method 활용 연습
from bs4 import BeautifulSoup

html_page = '''
<html>
    <body>
        <h1>제목 태그</h1>
        <p>웹 문서 연습</p>
        <p>원하는 자료 확인</p>
    </body>
</html>
'''
soup = BeautifulSoup(html_page, 'html.parser')
print(type(soup),'\n')

h1 = soup.html.body.h1
print("h1 : ", h1.string)

p1 = soup.html.body.p                   # 최초의 p tag만 잡음
print("p1 : ", p1.string)

p2 = p1.next_sibling.next_sibling       # next_sibling 한 번 하면 </p>로 감. 한 번 더 하면 다음 <p>로 감
print("p2 : ", p2.string)               # DOM을 이용함. 현실적으로 힘듦.


print("\n-*- find() method 이용 -*-")
html_page2 = '''
<html>
    <body>
        <h1 id='title'>제목 태그</h1>
        <p>웹 문서 연습</p>
        <p id='my' class='our'>원하는 자료 확인</p>
    </body>
</html>
'''
soup2 = BeautifulSoup(html_page2, 'html.parser')
# find(tag_name, attrs, recursive, string)
print(soup2.p, ' ', soup2.p.string)
print(soup2.find('p').string)                           # 최초의 p_tag 읽음
print(soup2.find('p', id='my').string)                  # p_tag 중 id 'my' 읽음

print(soup2.find(id='my').string)                       # id 'my' 읽음.
print(soup2.find(class_='our').string)                  # class_ 'our' 읽음.

print(soup2.find(attrs = {'id':'my'}).string)           # id 'my' 읽음.
print(soup2.find(attrs = {'class':'our'}).string)       # class 'our' 읽음.


print("\n-*- find_all(), findAll() method 이용 -*-")
html_page3 = '''
<html>
    <body>
        <h1 id='title'>제목 태그</h1>
        <p>웹 문서 연습</p>
        <p id='my' class='our'>원하는 자료 확인</p>
        <div>
            <a href='https://www.naver.com'>naver</a><br>
            <a href='https://www.daum.net'>daum</a>
        </div>
    </body>
</html>
'''
soup3 = BeautifulSoup(html_page3, 'html.parser')

print(soup3.find_all(['a']))
print(soup3.find_all(['a', 'p']))

links = soup3.find_all(['a'])
for i in links:
    href = i.attrs['href']
    text = i.string
    print(href, ' ', text)

print('\n정규 표현식 사용도 가능하다.')
import re
links2 = soup3.find_all(href=re.compile(r'^https'))
print(links2)

for k in links2:
    print(k.attrs['href'])

print('\n','-' * 10, 'bugs 사이트 음악 순위 읽기', '-' * 10)
import requests

url = 'https://music.bugs.co.kr/chart'
response = requests.get(url=url)
# print(response.text)

bugs_soup = BeautifulSoup(response.text, 'html.parser')

musics = bugs_soup.find_all('td', class_ = 'check')

for idx, musics in enumerate(musics):
    print(f"{idx+1}위)\t{musics.input['title']}")

