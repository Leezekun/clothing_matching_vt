import requests ##导入requests
import os
import time
import sqlite3

idb = sqlite3.connect("E:/polyvore7/item.db")
odb = sqlite3.connect("E:/polyvore7/outfit.db")
udb = sqlite3.connect("E:/polyvore7/user.db")
iodb = sqlite3.connect("E:/polyvore7/item_outfit.db")

icu = idb.cursor()
ocu = odb.cursor()
ucu = udb.cursor()
iocu = iodb.cursor()


icu.execute("create table item (iid varchar(10),iname varchar(100),kind varchar(100))")
ocu.execute("create table outfit (oid varchar(10) primary key,oname varchar(100),uid varchar(10),like varchar(10))")
ucu.execute("create table user (uid ivarchar(10) primary key,uname varchar(100),country varchar(100),link varchar(100),summary varchar(100))")
iocu.execute("create table item_outfit (oid varchar(10),iid varchar(10))")


idb.close()
odb.close()
udb.close()
iodb.close()

