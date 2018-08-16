import os
import random
import json
import string
import parser
import math
import csv
import socket
import webbrowser
import tkinter
import numpy

class gameKong():

    def __init__(self,type):
        games = {
            0 : "tiro",
            1 : "one",
            2 : "terror"
        }

        self.start = games.get(type,"nothing")
        self.len = lambda x : games.get(x)

    def start(self):

        card = numpy.array(["carro",""])
        rod = numpy.matrix()





if __name__ == "__main__" :

    game = gameKong(0)
    print(game.start)
    print(game.len(2))
    game.start()

    