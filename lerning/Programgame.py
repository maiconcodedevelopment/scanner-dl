from flask import Flask , session ,Request  ,Session ,request , sessions,  request , url_for
from game import Game , Cube
import numpy
import datetime
import csv

SESSION_TYPE = 'redis'

app = Flask(__name__)


@app.route('/')
def gameStart():

    listplayes = [10,50,24,54,34,65,4,12]

    value = listplayes[listplayes[6]:]

    return "Sim version game {0} ".format(value)


@app.route('/cube/')
def cubestart():
    cube , result = Cube(10,10,10,10,200).create()
    print(cube)
    print(result.encode())


@app.route('/machine/')
def startMachine():
    n = numpy.array([1,23,123,12,31,23,12,3])
    print(n)




@app.route('/start/')
def startGame():

    person = { 'name' : 'dinol {0}', 'wight' : float(233.4), 'height' : 100 , 'pes' : 200}

    person = dir(person)

    return person
