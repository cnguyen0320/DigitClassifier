#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import math

#Defines the node types
INPUT_NODE = 0
BIAS_HIDDEN = 1
HIDDEN = 2
BIAS_OUTPUT = 3
OUTPUT = 4

node_types = [INPUT_NODE, BIAS_HIDDEN, HIDDEN, BIAS_OUTPUT, OUTPUT]


class network():
    def __init__(trainingSet, numHiddenNodes, learning_rate, maxEpoch, seed, hiddenWeights, outputWeights):
        pass



"""
Class for neural network node
"""

class node():
    node_type = None
    parents = None
    inputValue = 0.0
    outputValue = 0.0
    outputGradient = 0.0
    delta = 0.0
    
    def __init__(self, node_type):
        assert node_type in node_types #make sure node specified is included as the type
        self.node_type = node_type
        
        self.arents = []
        
    def setInput(self, input_value):
        if self.node_type == INPUT_NODE:
            self.inputValue = input_value
    
    def calculateOutput(self):
        if (self.node_type == HIDDEN or self.node_type == OUTPUT):
            weightedSum = 0
            for parent in self.parents:
                weightedSum = weightedSum + parent.node.getOutput() * parent.weight
        
            if(self.node_type == HIDDEN):
                self.outputValue = max(0, weightedSum) #compute output for hidden layer node
                
            else:
                self.outputValue = math.exp(weightedSum) #compute output for output layer node
                
    def calculateSoftMax(self, total):
        if self.node_type == OUTPUT:
            self.outputValue = self.outputValue/total
    
    
    def calculateDeltaHidden(self, outputNodes):
        if self.node_type == HIDDEN:
            if self.outputValue == 0:
                self.delta = 0
            else:
                summation = 0.0
                
                for nwp in outputNodes:
                    for i in range (0, len(nwp.parents)):
                        if nwp.parents[i].node == self:
                            summation += nwp.parents[i].weight * nwp.delta
                
                self.delta = summation
    
    def updateWeights(self, learning_rate):
        if self.node_type == 2 or self.node_type == 4:
            for nwp in self.parents:
                deltaWeight = learning_rate * nwp.node.getOutput() * self.delta
                nwp.weight = nwp.weight + deltaWeight
    