import string
import csv
import numpy
import fileinput
import webbrowser
import requests

config = {"http" : "http://www.google.com"}

class Test(object):

    def __init__(self,value):
        self.value = value
        self.response = requests.get("http://www.google.com")

        print("este é o test")

class TestBase(Test):

    def __init__(self,value,values):

        print("este é o teste base {0}".format(values))
        super(TestBase, self).__init__(value)

    def read(self,*c):

        with open(c[0]) as file:
            print(file)

class Column(object):

    def __init__(self,name = None):
        if type(name) != str or name is None:
            raise Exception("is requried string")
        print(type(name))
        csv.reader(str="r")