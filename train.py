#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import NN

numAttributes = 256     # number of attributes per training image
numDigits = 10          # 0-9
numHiddenNodes = 10     # number of hidden nodes per layer in the NN
learning_rate = 0.5     # learning rate has to be between 0 and 1
maxEpoch = 0            # number of passes to do in the neural network training
seed = None             # use a seed (not None) to get consistent results from run to run
training_file = 'train.txt' # name of the file with training data


class Instance():
    attributes = list
    class_value = int
    
    def __init__(self, attribute_list, class_value):
        self.attributes=attribute_list
        self.value = class_value
    
    def add_attr(self, value):
        self.attributes.append(value)
        
    def getValue(self):
        return self.class_value


def parseTrainSet(training_file):
    
    #open the txt file with training data and read the liens
    file = open(training_file)
    training_set = file.readlines()
    file.close()
    
    inst_list = [] #create a list for the instance data
    
    #go through each training item and create an instance for it
    for line in training_set:
        values = line.split(" ")
        attributes = []
        for i in range(0, numAttributes): #populate the pixel values
            attributes.append(int(float(values[i])))
        
        digit = 0
        for x in range(numAttributes, len(values)): #find the class of the digit
            if int(values[x]) == 1:
                break
            digit = digit + 1
                
        inst = Instance(attributes, digit) #create the instance
        inst_list.append(inst) #add the instance to our list

    return inst_list
        

def randomizeWeights(seed = None):
    if not seed == None:
        random.seed(seed)
    
    hiddenWeights = []
    outputWeights = []
    
    #Randomizes weights for hidden nodes
    for x in range(0, numHiddenNodes):
        hiddenWeights.append([])
        for y in range(0, numAttributes+1):
            hiddenWeights[x].append(random.gauss(0,1)*.01)
    
    #Randomizes weights for output nodes
    for x in range(0, numDigits):
        outputWeights.append([])
        for y in range(0, len(hiddenWeights)+1):
            outputWeights[x].append(random.gauss(0,1)*.01)

    return(hiddenWeights, outputWeights)


if __name__ == "__main__":
    #Open the training data and save it to the training set
    trainingSet = parseTrainSet(training_file)
    
    #randomize the network weights
    (hiddenWeights, outputWeights) = randomizeWeights(seed)

    #create a neural network with the input parameters
    neural_net = NN.network(trainingSet, numHiddenNodes, learning_rate, maxEpoch, seed, hiddenWeights, outputWeights)

    neural_net.train();