import numpy as np
import requests
from sklearn.metrics import mean_squared_error

class deepLerning(object):

    def fit(self,input_data,weights):
        print("fit")

    def calcweights(self):
        weights = np.array([3,1])
        input_data = np.array([1,1])
        target = 10
        lerning_rate = 0.01

        preds = (input_data * weights).sum()
        error = target - preds
        print(error)

        #gradient decest
        gradient = 2 * input_data * error

        weights_updade = weights - lerning_rate * gradient
        preds_updade = (weights_updade * input_data).sum()
        print(preds_updade)



inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
marc = np.array([0,0,0,1])
weights = np.array([0.0,0.0])
lerning_rate = 0.1



def sum(input,weight):
    return np.dot(input,weight)

def sigmoid(sum):
    return 1 / (1 + np.exp(-sum))



def stepFunction(sum):
    if(sum >= 1):
        return 1
    return 0

def updadeWeights():
    print("*")
    #PESO(N +1) - PESO(N) + (TAXAAPRENDIZAGEM * ENTRADA * ERRO)

def calcUp(regist):
    s = regist.dot(weights)
    return stepFunction(s)

def train():
    errostotal = 1
    while (errostotal != 0):
        errostotal = 0
        for i in range(len(marc)):
            saidaCalculada = calcUp(np.asarray(inputs[i]))
            erro = abs(marc[i] - saidaCalculada)
            errostotal += erro
            for j in range(len(weights)):
                weights[j] = weights[j] + (lerning_rate * inputs[i][j] * erro )
                print("Peso atualizado :" + str(weights[j]))
        print("Total de erros : " + str(errostotal))


# train()
# deepLerning().calcweights()
print(sigmoid(50))
print(np.exp(0))
# total = sum(inputs,weights)
# step = stepFunction(total)

# print(total)
# print(step)
import cv2
cv2.KNearest()
#context
# descoverta de novos remedios , entendimento de lunguagem natual , carros autonomos , recnhecimento facil , cura para doenças , bolsa de valores , encontrar soluções para controle de tráfego