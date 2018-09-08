import numpy as np
import keras
import tensorflow


def predict_with_network(input_data_row,weights):

    node_0_0_value = (input_data_row * weights["node_0_0"]).sum()
    node_0_0_output = relu(node_0_0_value)
    
    node_0_1_value = (input_data_row * weights["node_0_1"]).sum()
    node_0_1_output = relu(node_0_1_value)

    hidden_0_outputs = np.array([node_0_0_value,node_0_1_output])

    node_1_0_value = (hidden_0_outputs * weights["node_1_0"]).sum()
    node_1_0_output = relu(node_1_0_value)

    node_1_1_value = (hidden_0_outputs * weights["node_1_1"]).sum()
    node_1_1_output = relu(node_1_1_value)

    hidden_1_outputs = np.array([node_1_0_output,node_1_1_output])

    model_output = (hidden_1_outputs * weights["output"]).sum()

    return model_output



#function relu active linear
def relu(input):
    return max(0,input)

input_data = np.array([[3,5],[1,-1],[0,0],[8,4]])

#weights
weight = { "node_0_0" : np.array([2,4]), 
           "node_0_1" : np.array([4,-5]),
           "node_1_0" : np.array([-1,2]),
           "node_1_1" : np.array([1,2]),
           "output" : np.array([2,7]) }

results = []
for input_data_row in input_data:
    results.append(predict_with_network(input_data_row,weight))

print(results)


#when you train model , the neural network gets weights
#that find the relevant patterns to make better predictions

#example as seen by LINEAR regression
# age
# income
# children
# accounts
# bank balance
# retirement status
# number of transactions
# model with no interactions

# neural networks account for interactions really well

# deep learning uses especially powerful neural networks

# text
# images
# videos
# autdio
# source code

# coruse structure

# first two chapters focus on conceptual knmowledge

# debug and tune deep learning models on conventional prediction problems

# lay the founcdation for progressiong toward modern applications

# this will pay off iin the third and foruth chapters

# build deep leraning models with keras

# Comparing neural network models to classical regression models

# Which of the models in the diagrams has greater ability to account for interactions?

# Forward propagation

# bank transactions example
# make predictions based on
# number of existing accounts

# add process
# dot product
# for one data point at a time
# output is the prediction for that data point

# Coding the forward propagation algorithm

# In this exercise, you'll write code to do forward propagation (prediction) for your first neural network:

# The weights feeding into the output node are available in weights['output'].

# linear vs nonlinear functions

# applied to node inputs to product node output
# improvind our neural network

# Applying the network to many observations/rows of data

# Deeper networks

# calculate with relu activation function

# deep networks internally build representations of patterns in the data
# pantilly replace the need for feature engi

# subsequent layers build increasingly sophisticated representations of raw data

# modeler doesnt nedd to specify the interactions

# when you train model , the neural network gets weights
# that find the relevant patterns to make better predictions


#Forward propagation in a deeper network

# You now have a model with 2 hidden layers. The values for an input data point are shown inside the input nodes. The weights are shown on the edges/lines. What prediction would this model make on this data point?

# Assume the activation function at each node is the identity function. That is, each node's output will be the same as its input. So the value of the bottom node in the first hidden layer is -1, and not 0, as it would be if the ReLU activation function was used.

# Multi-layer neural networks

# In this exercise, you'll write code to do forward propagation for a neural network with 2 hidden layers. Each hidden layer has two nodes. The input data has been preloaded as input_data. The nodes in the first hidden layer are called node_0_0 and node_0_1. Their weights are pre-loaded as weights['node_0_0'] and weights['node_0_1'] respectively.

# The nodes in the second hidden layer are called node_1_0 and node_1_1. Their weights are pre-loaded as weights['node_1_0'] and weights['node_1_1'] respectively.

# We then create a model output from the hidden nodes using weights pre-loaded as weights['output']

# the need for optimization

# a baseline neural network
# actural value of target

# error predicted - actual = -4

# making accurate predictions gets harder with more points
# at any set of weights , there are many values of the erros
# corresponding to the many points we make predictions for

# loss function

# making accurate predictions gets harder with more points
# at any set of weights , there are many values of the erros
# corresponding to the many points we make predictions for

# loss function

# aggregates errros in predictions from many data points into single number

# measure of model predictive performance
# squared error loss function

# actual

# lower loss function value means a better model
# goal find the weights that give the lowest value for the loss function
# gradient descent

# imagine you are in a pitch dark field
# want to find the lowest point
# feel the ground to see how it shopes
# take a small step downhill
# repet until it is uphill in every direction

# start at random poin
# until tou are somewhere flat
# find the slope
# take a step downhill

# optimizing a model with a single weight
# Calculating model errors

# For the exercises in this chapter, you'll continue working with the network to predict transactions for a bank.

# What is the error (predicted - actual) for the following network when the input data is [3, 2] and the actual value of the target (what you are trying to predict) is 5? It may be helpful to get out a pen and piece of paper to calculate these values.

# Understanding how weights change model accuracy

# Imagine you have to make a prediction for a single data point. The actual value of the target is 7. The weight going from node_0 to the output is 2, as shown below. If you increased it slightly, changing it to 2.01, would the predictions become more accurate, less accurate, or stay the same?
# More accurate.
# press 1
# Less accurate.
# press 2
# Stay the same.