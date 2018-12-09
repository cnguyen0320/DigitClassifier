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
    seed = parameters['network']['seed']
    
    return training_file, numAttributes, numDigits,numHiddenNodes,learning_rate,maxEpoch,seed

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
                
        inst = NN.Instance(attributes, digit) #create the instance
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
    trainingSet = parseTrainSet(training_file)
    
    #randomize the network weights
    (hiddenWeights, outputWeights) = randomizeWeights(seed)
    
    #create a neural network with the input parameters
    neural_net = NN.Network(trainingSet,numAttributes,numDigits, numHiddenNodes, learning_rate, maxEpoch, hiddenWeights, outputWeights, seed)

    neural_net.train(save_weights = save, loss_output = debug);
    
    #%%
    random_index = math.floor(random.random()*len(trainingSet))
    
    inst = trainingSet[random_index]
    
    print(inst.value)
    
    print("The detected digit is a", neural_net.predict(inst))