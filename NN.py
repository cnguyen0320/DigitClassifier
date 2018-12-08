#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import math, random
from Instance import Instance

#Defines the node types
INPUT_NODE = 0
BIAS_HIDDEN = 1
HIDDEN = 2
BIAS_OUTPUT = 3
OUTPUT = 4

node_types = [INPUT_NODE, BIAS_HIDDEN, HIDDEN, BIAS_OUTPUT, OUTPUT]


class Network():
    """
    Class for the neural network 
    
    It can be used to train a training set or predict a score given an input
    """
    
    learningRate = None
    maxEpoch = None
    seed = None
    inputNodes = None
    hiddenNodes = None
    outputNodes = None
    
    def __init__(self, trainingSet, number_of_attributes, number_of_values, number_of_hidden_nodes, learningRate, maxEpoch, hiddenWeights, outputWeights, seed = None):
        self.trainingSet = trainingSet
        self.numAttributes = number_of_attributes
        self.numValues = number_of_values
        self.numHiddenNodes = number_of_hidden_nodes
        self.learningRate = learningRate
        self.maxEpoch = maxEpoch
        self.hiddenWeights = hiddenWeights
        self.outputWeights = outputWeights
        self.seed = seed
        random.seed(seed)
        
        #Create input layer nodes
        self.inputNodes = []
        for x in range(0, self.numAttributes):
            self.inputNodes.append(Node(INPUT_NODE))
        
        #Creates the bias node to the hidden layer
        self.inputNodes.append(Node(BIAS_HIDDEN))
        
        #Create the hidden layer nodes
        self.hiddenNodes = []
        for x in range(0, self.numHiddenNodes):
            node = Node(HIDDEN)
            for y in range(0, len(self.inputNodes)):
                nwp = NodeWeightPair(self.inputNodes[y], hiddenWeights[x][y])
                node.parents.append(nwp)
            self.hiddenNodes.append(node)
        
        
        #Create the bias node to the output layer
        self.hiddenNodes.append(Node(BIAS_OUTPUT))
        
        #Create output layer nodes
        self.outputNodes = []
        for x in range(0, self.numValues):
            node = Node(OUTPUT)
            for y in range(0, len(self.hiddenNodes)):
                nwp = NodeWeightPair(self.hiddenNodes[y], outputWeights[x][y])
                node.parents.append(nwp)
            self.outputNodes.append(node)

    def train(self):
        #train for max epochs
        for epoch in range(0, self.maxEpoch):
            
            #randomize the training set order
            random.shuffle(self.trainingSet)
            
            #for each training instance, compute forward and backward pass
            for instance in self.trainingSet:
                self.forwardPass(instance)
                self.backwardPass(instance)
        
        for
        
        print('training complete for', self.maxEpoch, 'epochs')
        
            

    def forwardPass(self, instance):
        """Computes forward pass to get outputs for a given input"""
        #set input nodes using instance attributes
        for i in range(0,self.numAttributes):
            self.inputNodes[i].setInput(instance.attributes[i])
        
        #compute output for hidden layer
        for node in self.hiddenNodes:
            node.calculateOutput()
            
        #compute output for output layer and accumulate for softmax
        softmaxTotal = 0.0
        for node in self.outputNodes:
            node.calculateOutput()
            softmaxTotal = softmaxTotal + node.outputValue
            
        for node in self.outputNodes:
            node.calculateSoftMax(softmaxTotal)
        
    def backwardPass(self, instance):
        """Computes back propagation to change weights during training"""
        
        #calculate deltas for output nodes
        for i in range(0, self.numValues):
            #if output node matches the type of the instance, it should be true
            if i == instance.value:
                self.outputNodes[i].calculateDeltaOutput(1)

            #the node should not be on
            else:
                self.outputNodes[i].calculateDeltaOutput(0)
                
        #calculate deltas for hidden layer nodes
        for node in self.hiddenNodes:
            node.calculateDeltaHidden(self.outputNodes)
            
        #Apply changes to the weights
        for node in self.outputNodes + self.hiddenNodes:
            node.updateWeight(self.learningRate)



class NodeWeightPair():
    """
    Class that stores connection between layers
    
    Contains the parent node and the weight of the connection
    """
    
    def __init__(self, node, weight):
        self.node = node
        self.weight = weight

class Node():
    """
    Class for neural network node
    
    Each node has a type, list of parents (if not an input node), an input, output, and delta
    """
    
    node_type = None
    parents = None
    inputValue = 0.0
    outputValue = 0.0
    delta = 0.0
    
    def __init__(self, node_type):
        
        """Creates a node with specified type"""
        assert node_type in node_types #make sure node specified is included as the type
        self.node_type = node_type
        self.parents = []
        
    def setInput(self, input_value):
        """Sets the input value for input nodes"""
        if self.node_type == INPUT_NODE:
            self.inputValue = input_value
    
    
    
    def calculateOutput(self):
        """Computes output for the hidden or output layer nodes"""
        
        if (self.node_type == HIDDEN or self.node_type == OUTPUT):
            weightedSum = 0
            for parent in self.parents:
                weightedSum = weightedSum + parent.node.outputValue * parent.weight
        
            #compute output for hidden layer node
            if(self.node_type == HIDDEN):
                self.outputValue = max(0, weightedSum) 
                
            #compute output for output layer node
            else:
                self.outputValue = math.exp(weightedSum) 
                
                
    
    def calculateSoftMax(self, total):
        """Computes softmax for the output layer nodes"""
        if self.node_type == OUTPUT:
            self.outputValue = self.outputValue/total
    
    
    def calculateDeltaOutput(self, targetOutput):
        if self.node_type == OUTPUT:
            self.delta = targetOutput - self.outputValue
    
    
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
    
    
    
    def updateWeight(self, learning_rate):
        """
        Updates the weights of a node. Used for 
        """
        if self.node_type == 2 or self.node_type == 4:
            for nwp in self.parents:
                deltaWeight = learning_rate * nwp.node.outputValue * self.delta
                nwp.weight = nwp.weight + deltaWeight
    