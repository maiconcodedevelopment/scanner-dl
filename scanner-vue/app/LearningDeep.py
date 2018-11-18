import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import  style
from sklearn.metrics import mean_squared_error

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

# atualizações podem não melhorar o modelo de forma significativa
# simultaneamente otimizando milhares de parâmetros com relacionamentos complexos
# atualizações muito pequenas (se a taxa de aprendizado for baixa) ou muito grande (se a taxa de aprendizado for alta)
# 2 * x * (y-xb)


#convexo não tem o minimo local , #não convexo ele tem o minimos globals

def stefunction(sum):
    return  1 if (sum >= 1 ) else 0

#y = 1/ 1+e-x
def sigmoid(sum):
    return 1 / (1 + np.exp(-sum))

#d = y*(1 - y)
def sigmoidDerivada(sum):
    return sum * (1 - sum)

# tangente iperbolica
def tahnFunctin(sum):
    return (np.exp(sum) - np.exp(-sum)) / (np.exp(sum)) + np.exp(-sum)

def reluFunction(sum):
    return max(0,sum)

def linearFunction(sum):
    return sum

def tanhFunction(sum):
    return (np.exp(sum) - np.exp(-sum)) / (np.exp(sum) + np.exp(-sum))

def softmaxFunction(sum):
    ex = np.exp(sum)
    return ex / ex.sum()

moment = 1
lerning_rate = 0.1

def animate(v):
    input_data_1 = np.array([[0, 1, 1, 0]])
    input_data = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]])

    weight_1 = np.array([[0.432, 0.123, 0.654, 0.342], [0.341, 0.123, 0.654, 0.235], [0.543, 0.542, 0.123, 0.231],
                         [0.234, 0.123, 0.776, 0.645]])
    weight_2 = np.array([[0.653], [0.234], [0.341], [0.341]])

    saidas = np.array([[0], [1], [1], [0]])

    x = []
    y = []

    for i in range(100000000):

        camada_1 = np.dot(input_data,weight_1)
        camada_1_1 = sigmoid(camada_1)

        camada_2 = np.dot(camada_1_1,weight_2)
        camada_2_2 = sigmoid(camada_2)

        erros = saidas - camada_2_2
        mean_absolute = np.mean(erros)

        print("Mean Errors :{0}".format(np.abs(mean_absolute)))

        x.append(i)
        y.append(np.abs(mean_absolute))

        ax1.clear()
        ax1.plot(x,y)

        derivada_output = sigmoidDerivada(camada_2_2)  # derivada ativação (sigmoid)
        delta_output = erros * derivada_output #delta de saida

        # derivadasigmoide * peso * deltasaida
        matrix_weight_2 = weight_2.T
        delta_x_weight = delta_output.dot(matrix_weight_2)
        delta_camad_hidden = delta_x_weight * sigmoidDerivada(camada_1_1)

        # (peso * memonto) + (entrada + delta + taxa de aprendizagem)
        camada_hidden_transport = camada_1_1.T
        weight_new = camada_hidden_transport.dot(delta_output)
        weight_2 = (weight_2 * moment) + (weight_new * lerning_rate)

        camada_input_transport = input_data.T
        weight_new_1 = camada_input_transport.dot(delta_camad_hidden)
        weight_1 = (weight_1 * moment) + (weight_new_1 * lerning_rate)


ani = animation.FuncAnimation(fig,animate,interval=10000)
plt.show()



