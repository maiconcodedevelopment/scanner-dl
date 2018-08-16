import math
import fileinput

class Cube(object):

    def __init__(self,x,y,z,width,height):

        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.z = z

    def create(self):

        cube = self.width ** self.height
        return cube , "sim este é o cube"

class Person (object):

    def __init__(self,**caracters):
        self.person = caracters


class Game(Person):

    def __init__(self,person = dict(),name = ""):
        super(Game, self).__init__(person,name)

        self.number = ['m','k','j','p','w']
        self.active = ['_','_','_','_','_']
        self.quantity = int()
        self.errors = int()


    def start(self):

        adiv = len(self.number)
        adivactive = []


        while(adiv != self.quantity):
            value = input("what caracter ?:")

            if value in self.number:
                adivactive.append(value)
                self.quantity = self.quantity + 1
                print(adivactive)
                print("yes good !")
            else :
                self.quantity = self.quantity + 1
                self.errors = self.errors + 1

            print(self.quantity)
            print(self.errors)

        bith = self.quantity - self.errors

        print("end***** {0} {1}" .format(bith,"bithday"))