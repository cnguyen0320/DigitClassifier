#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import NN
import train
import math, random, json, sys

def loadWeights(filename):
    """Loads the saved weights and creates lists with them"""
    file = open(filename, 'r')
    text = file.readlines()
    file.close()
    
    weights = []
    for line in text:
        if '#' not in line:
            values = line.split(" ")
            values.remove('\n')
            if len(values)>0:
                weights.append(values)

    return weights

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

def test(network, trainingSet):
    count = 0
    for inst in trainingSet:
        if inst.value == network.predict(inst):
            count = count + 1
    return count/len(trainingSet)

if __name__ == '__main__':
    
    try:
        #get filenames from the command line argument and load them
        json_file = str(sys.argv[1])
        hiddenfile = str(sys.argv[2])
        outputfile = str(sys.argv[3])
        hiddenWeights = loadWeights(hiddenfile)
        outputWeights = loadWeights(outputfile)
        training_file, numAttributes, numDigits,numHiddenNodes,learning_rate,maxEpoch,seed = loadJSON(json_file)
        
    except:
        print('Error loading. usage: python train.py <parameter_file> <hiddenweights_file> <outputweights_file' )
        exit(1)
    
    
    #Creates a network using the parameters set in train.py and the loaded hidden and output weights
    network = NN.Network(None, None, numAttributes, numDigits, numHiddenNodes, learning_rate, maxEpoch, hiddenWeights, outputWeights)
    

    inst_list = train.parseDataSet(training_file, numAttributes)    
    random_index = math.floor(random.random()*len(inst_list))    
    inst = inst_list[random_index]
    
    print("Known value:", inst.value)
    
    print("The detected digit is a", network.predict(inst))
    
    print("success rate: ", test(network, inst_list))