import requests ##导入requests
import os
import time
import sqlite3

idb = sqlite3.connect("E:/polyvore/polyvore8/item.db")
odb = sqlite3.connect("E:/polyvore/polyvore8/outfit.db")
udb = sqlite3.connect("E:/polyvore/polyvore8/user.db")
tdb = sqlite3.connect("E:/polyvore/polyvore8/top.db")
bdb = sqlite3.connect("E:/polyvore/polyvore8/bottom.db")
sdb = sqlite3.connect("E:/polyvore/polyvore8/shoes.db")
iodb = sqlite3.connect("E:/polyvore/polyvore8/item_outfit.db")
todb = sqlite3.connect("E:/polyvore/polyvore8/top_outfit.db")
bodb = sqlite3.connect("E:/polyvore/polyvore8/bottom_outfit.db")
sodb = sqlite3.connect("E:/polyvore/polyvore8/shoes_outfit.db")
uldb = sqlite3.connect("E:/polyvore/polyvore8/user_link.db")

icu = idb.cursor()
ocu = odb.cursor()
ucu = udb.cursor()
tcu = tdb.cursor()
bcu = bdb.cursor()
scu = sdb.cursor()
iocu = iodb.cursor()
tocu = todb.cursor()
bocu = bodb.cursor()
socu = sodb.cursor()
ulcu = uldb.cursor()


'''
icu.execute("create table item (iid varchar(10),iname varchar(100),kind varchar(100))")
ocu.execute("create table outfit (oid varchar(10) primary key,oname varchar(100),uid varchar(10),like varchar(10))")
ucu.execute("create table user (uid ivarchar(10) primary key,uname varchar(100),country varchar(100),link varchar(100),summary varchar(100))")
iocu.execute("create table item_outfit (oid varchar(10),iid varchar(10))")
'''

#icu.execute("insert into item values('sad','sas','d')")
#idb.commit()

for i in socu.execute("select * from shoes_outfit").fetchall():
    print(i)

idb.close()
odb.close()
udb.close()
tdb.close()
bdb.close()
sdb.close()
iodb.close()
todb.close()
bodb.close()
sodb.close()
uldb.close()





