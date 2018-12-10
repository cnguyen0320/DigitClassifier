#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random, json, sys
import NN
import math

def loadJSON(filename):
    data = open(filename)
    parameters = json.load(data)
    data.close()
    
    training_file = parameters['data']['training_data']
    numAttributes = parameters['data']['num_attributes']
    numDigits = parameters['data']['num_classes']
    numHiddenNodes = parameters['network']['num_hidden_nodes']
    learning_rate = parameters['network']['learning_rate']
    maxEpoch = parameters['network']['max_epochs']
    try:
        seed = int(parameters['network']['seed'])
    except:
        seed = None
    
    return training_file, numAttributes, numDigits,numHiddenNodes,learning_rate,maxEpoch,seed

def parseDataSet(training_file, num_attributes):
    
    #open the txt file with training data and read the liens
    file = open(training_file)
    training_set = file.readlines()
    file.close()
    
    inst_list = [] #create a list for the instance data
    
    #go through each training item and create an instance for it
    for line in training_set:
        values = line.split(" ")
        attributes = []
        for i in range(0, num_attributes): #populate the pixel values
            attributes.append(int(float(values[i])))
        
        digit = 0
        for x in range(num_attributes, len(values)): #find the class of the digit
            if int(values[x]) == 1:
                break
            digit = digit + 1
                
        inst = NN.Instance(attributes, digit) #create the instance
        inst_list.append(inst) #add the instance to our list

    return inst_list
        

def randomizeWeights(num_hidden, num_attributes, num_values):    
    hiddenWeights = []
    outputWeights = []
    
    #Randomizes weights for hidden nodes
    for x in range(0, num_hidden):
        hiddenWeights.append([])
        for y in range(0, num_attributes+1):
            hiddenWeights[x].append(random.gauss(0,1)*.01)
    
    #Randomizes weights for output nodes
    for x in range(0, num_values):
        outputWeights.append([])
        for y in range(0, len(hiddenWeights)+1):
            outputWeights[x].append(random.gauss(0,1)*.01)

    return(hiddenWeights, outputWeights)

def test(network, trainingSet):
    count = 0
    for inst in trainingSet:
        if inst.value == network.predict(inst):
            count = count + 1
    return count/len(trainingSet)

def divideDataSet(data_set, percent_to_train, seed = None):
    random.seed(seed)
    
    train = []
    tune = []
    for instance in data_set:
        if random.uniform(0,1) < percent_to_train:
            train.append(instance)
        else:
            tune.append(instance)
    return (train, tune)


if __name__ == "__main__":

    save = True
    debug = False    
    
    try:
        #load in the json file from the command line argument and load it
        json_file = str(sys.argv[1])
        training_file, numAttributes, numDigits,numHiddenNodes,learning_rate,maxEpoch,seed = loadJSON(json_file)
        if '-d' in sys.argv:
            debug = True
        if '-s' in sys.argv:
            save = False
        
    except:
        print('Error loading JSON. usage: python train.py <parameter_file.json>')
        print("OPTIONAL: -d: print losses, -s: don't save weights")
        exit(1)
    
    #Open the training data and save it to the training set
    dataSet = parseDataSet(training_file, numAttributes)
    
    trainingSet, tuningSet = divideDataSet(dataSet, 0.5, seed)
    
    #randomize the network weights
    (hiddenWeights, outputWeights) = randomizeWeights(numHiddenNodes,numAttributes,numDigits)
    
    #create a neural network with the input parameters and train it
    network = NN.Network(trainingSet, tuningSet, numAttributes,numDigits, numHiddenNodes, learning_rate, maxEpoch, hiddenWeights, outputWeights, seed)
    network.train(save, debug);
    
            