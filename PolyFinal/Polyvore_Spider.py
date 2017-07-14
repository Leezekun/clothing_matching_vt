import requests ##导入requests
from bs4 import BeautifulSoup ##导入bs4中的BeautifulSoup
from selenium import webdriver
import os
import time
import sqlite3


headers = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1"}##浏览器请求头（大部分网站没有这个请求头会报错、请务必加上哦）
all_url = 'http://www.polyvore.com/cgi/search.sets?item_count.from=4&item_count.to=10'  ##开始的URL地址

'''
start_html = requests.get(all_url,  headers=headers)

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches",["ignore-certificate-errors"])
driver = webdriver.Chrome(chrome_options=options)

driver.get(all_url)
driver.maximize_window()

i = 0

while i < 2:
#driver.execute_script("window.scrollBy(0,10000)")
 js = "window.scrollTo(0,document.body.scrollHeight)"
 driver.execute_script(js)
 time.sleep(1)
 i = i + 1
 print (i)

start_html1 = driver.page_source
print(start_html1)

'''

f = open('E:/polyvore7/htmlcopy.txt','r',encoding='utf-8')
start_html = f.read()
f.close()

idb = sqlite3.connect("E:/polyvore7/item.db")
odb = sqlite3.connect("E:/polyvore7/outfit.db")
udb = sqlite3.connect("E:/polyvore7/user.db")
iodb = sqlite3.connect("E:/polyvore7/item_outfit.db")


icu = idb.cursor()
ocu = odb.cursor()
ucu = udb.cursor()
iocu = iodb.cursor()


f1 = open('E:/polyvore7/user.txt','a+',encoding='utf-8')
f2 = open('E:/polyvore7/outfit.txt','a+',encoding='utf-8')
f3 = open('E:/polyvore7/item.txt','a+',encoding='utf-8')
f4 = open('E:/polyvore7/item_outfit.txt','a+',encoding='utf-8')

os.chdir("E:\polyvore7\image") ##切换到上面创建的文件夹

#Soup = BeautifulSoup(start_html.text, 'lxml')
Soup = BeautifulSoup(start_html, 'lxml')
grid = Soup.find_all('ul',class_='layout_n grid grid_6')

for g in grid:

  all_li_12 = g.find_all('li',class_='size_l2')
  all_li_12_last = g.find_all('li',class_='size_l2 last')
  all_li_12_row = g.find_all('li',class_='size_l2 last_row')
  all_li_12_last_row = g.find_all('li',class_='size_l2 last last_row')
  for li in all_li_12 or all_li_12_last or all_li_12_row or all_li_12_last_row:
        t = li.find('div', class_='title')
        c = li.find('div', class_='createdby')
        l = li.find('span', class_='fav_count')
        at = t.find("a")
        ac = c.find('a')

        title = at.get_text() #取出a标签的文本

        link = at['href']
        create = ac.get_text()
        user_href = ac['href']
        user_link =user_href
        uh = requests.get(user_link, headers=headers)
        uh_Soup = BeautifulSoup(uh.text, 'lxml')
        info = uh_Soup.find('div',class_="user_info")
        country = info.find('div',class_='meta').get_text()
        try:
           ins_soup = info.find('div',class_='user_links clearfix').find('a')
           ins = ins_soup['href']
        except:
           ins = ''

        more = uh_Soup.find('ul',class_="activity_summary").find_all('li')
        summary = ''
        for x in more:
                summary = summary + x.get_text()

        like = l.get_text()

#        if create in ua:
#            uid = ua.index(create)
#        else:
#            f1.write(str(uc)+" "+create+" "+country+" "+ins+" "+summary+"\n")
#            ua.append(create)
#            uid = uc
#            uc = uc + 1

        uid_sql = ucu.execute("select uid from user where uname='"+create.replace("'",'')+"'").fetchone()
        if uid_sql is not None:
            uid = str(uid_sql).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
        else:
            uc = ucu.execute("select count(*) from user").fetchone()
            uid = str(uc).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
            f1.write(str(uid) + " " + create + " " + country + " " + ins + " " + summary + "\n")
            ucu.execute("insert into user values('"+str(uid)+"','"+create.replace("'",'')+"','"+country.replace("'",'')+"','"+ins.replace("'",'')+"','"+summary.replace("'",'')+"')")
            udb.commit()

        oid_sql = ocu.execute("select oid from outfit where oname='" + title.replace("'", '') + "'").fetchone()
        if oid_sql is not None:
            oid = str(oid_sql).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
        else:
            oc = ocu.execute("select count(*) from outfit").fetchone()
            oid = str(oc).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
            f2.write(str(oid) + " " + title + " " + str(uid) + " " + like + "\n")
            ocu.execute("insert into outfit values('"+str(oid)+"','"+title.replace("'",'')+"','"+str(uid)+"','"+like.replace("'",'')+"')")
            odb.commit()



        link = all_url + link
        html = requests.get(link, headers=headers)
        html_Soup = BeautifulSoup(html.text, 'lxml')
        item = html_Soup.find('ul',class_='layout_grid grid_5 mod_inline_save clearfix ').find_all('div',class_="grid_item hover_container type_thing span1w span1h")
        for i in item:
            a = i.find('div',class_='main').find('a')
            href = a['href']
            page_url = link + href
            img_html = requests.get(page_url, headers=headers)
            img_Soup = BeautifulSoup(img_html.text, 'html.parser')
            img_url = a.find('img')['src']
            try:
                kind = img_Soup.find('div', id='body').find('div',class_='page thing').find('div',class_='clearfix').find('div',id='right').find_all('span',itemprop='title')
                kind_des = ''

                for k in kind:
                    kind_des = kind_des + '>'+ k.get_text()

                text_url = img_Soup.find('div', id='body').find('div',class_='page thing').find('div',class_='clearfix').find('div',id='right').find('h1')
                name = text_url['title']

                img = requests.get(img_url, headers=headers)

#            if name in ia:
#                iid = ia.index(name)
#            else:
#                iid = ic
#                f3.write(str(iid) +" " + name + " " + kind_des+ " "+"\n")
#                ic = ic + 1

                iid_sql = icu.execute("select iid from item where iname='" + name.replace("'",'') + "'").fetchone()
                if iid_sql is not None:
                    iid = str(iid_sql).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
                else:
                    ic = icu.execute("select count(*) from item").fetchone()
                    iid = str(ic).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
                    f3.write(str(iid) + " " + name + " " + kind_des + " " + "\n")
                    icu.execute("insert into item values('" + str(iid) + "','" + name.replace("'",'') + "','" + kind_des.replace("'",'') +"')")
                    idb.commit()
                    f = open(str(iid) + '.jpg', 'ab')
                    f.write(img.content)
                    f.close()

                ioid_sql = ucu.execute("select oid from item_outfit where iid='" + str(iid).replace("'", '') + "'").fetchone()
                if ioid_sql is None:
                    f4.write(str(oid)+" "+str(iid)+"\n")
                    iocu.execute("insert into item_outfit values('" + str(oid) + "','" + str(iid) + "')")
                    iodb.commit()
                else:
                    ioid_sql1 = ucu.execute("select oid from item_outfit where iid='" + str(iid).replace("'", '') + "'").fetchall()
                    if ioid_sql  in ioid_sql:
                        print("item_outfit has existd")
                    else:
                        f4.write(str(oid) + " " + str(iid) + "\n")
                        iocu.execute("insert into item_outfit values('" + str(oid) + "','" + str(iid) + "')")
                        iodb.commit()
            except:
                print("faild to get item")



icu = idb.close()
ocu = odb.close()
ucu = udb.close()
iocu = iodb.close()



f1.close()
f2.close()
f3.close()
f4.close()