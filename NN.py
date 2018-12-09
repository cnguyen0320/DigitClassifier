#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import math, random

class Network():
    """
    Class for the neural network 
    
    It can be used to train a training set or predict a score given an input
    """
        
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
            self.inputNodes.append(Node(Node.INPUT_NODE))
        
        #Creates the bias node to the hidden layer
        self.inputNodes.append(Node(Node.BIAS_HIDDEN))
        
        #Create the hidden layer nodes
        self.hiddenNodes = []
        for x in range(0, self.numHiddenNodes):
            node = Node(Node.HIDDEN)
            for y in range(0, len(self.inputNodes)):
                nwp = NodeWeightPair(self.inputNodes[y], hiddenWeights[x][y])
                node.parents.append(nwp)
            self.hiddenNodes.append(node)
        
        
        #Create the bias node to the output layer
        self.hiddenNodes.append(Node(Node.BIAS_OUTPUT))
        
        #Create output layer nodes
        self.outputNodes = []
        for x in range(0, self.numValues):
            node = Node(Node.OUTPUT)
            for y in range(0, len(self.hiddenNodes)):
                nwp = NodeWeightPair(self.hiddenNodes[y], outputWeights[x][y])
                node.parents.append(nwp)
            self.outputNodes.append(node)



    def train(self, save_weights = True, loss_output = False):
        """Computes forward and backward passes for the NN to set the weights"""
        #train for max epochs
        for epoch in range(0, self.maxEpoch):
            
            #randomize the training set order
            random.shuffle(self.trainingSet)
            totalLoss = 0.0
            
            #for each training instance, compute forward and backward pass
            for instance in self.trainingSet:
                self.forwardPass(instance)
                self.backwardPass(instance)
            
            if loss_output:
                #for each instance, compute a loss
                for inst in self.trainingSet:
                    totalLoss = totalLoss + self.loss(inst)
                
                totalLoss = totalLoss/len(self.trainingSet)
                
                print("Epoch", epoch, "complete with loss:", format(totalLoss, 'e'))
            
        #save the weights into the file
        if save_weights: 
            self.saveWeights()
        
        
    def predict(self, instance):
        """Returns the prediction for a given input"""
        self.forwardPass(instance)
        
        #go through the outputs and find the maximum index
        maxIndex = 0
        maxVal = float("-inf")
        
        for i in range(0, len(self.outputNodes)):
            current = self.outputNodes[i].outputValue
            
            if current > maxVal:
                maxIndex = i
                maxVal = current
        
        return maxIndex



    def loss(self, instance):
        sum = 0
        self.forwardPass(instance)
        for x in range(0, len(self.outputNodes)):
            logPredicted = math.log(self.outputNodes[x].outputValue)
            if x == instance.value:
                sum = sum + logPredicted
        return sum

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


    def saveWeights(self):
            
        file = open('hiddenweights.txt', 'w')
        file.write('#hidden: ' + str(len(self.hiddenNodes)) +'\n')
        file.write('#output: ' + str(len(self.outputNodes)) +'\n')
        for node in self.hiddenNodes:
            for nwp in node.parents:
                file.write(str(nwp.weight))
                file.write(' ')
            file.write('\n')
        file.close()
        
        file = open('outputweights.txt', 'w')
        file.write('#hidden: ' + str(len(self.hiddenNodes)) +'\n')
        file.write('#output: ' + str(len(self.outputNodes)) +'\n')
        for node in self.outputNodes:
            for nwp in node.parents:
                file.write(str(nwp.weight))
                file.write(' ')
            file.write('\n')
        file.close()


class NodeWeightPair():
    """
    Class that stores connection between layers
    
    Contains the parent node and the weight of the connection
    """
    
    def __init__(self, node, weight):
        self.node = node
        self.weight = float(weight)

class Node():
    #Defines the node types
    INPUT_NODE = 0
    BIAS_HIDDEN = 1
    HIDDEN = 2
    BIAS_OUTPUT = 3
    OUTPUT = 4
    
    node_types = [INPUT_NODE, BIAS_HIDDEN, HIDDEN, BIAS_OUTPUT, OUTPUT]
    
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
        assert node_type in Node.node_types #make sure node specified is included as the type
        self.node_type = node_type
        self.parents = []
        
        #Set the input/output of all bias nodes to 1
        if self.node_type == Node.BIAS_OUTPUT or self.node_type == Node.BIAS_HIDDEN:
            self.outputValue = 1.0
            self.inputValue = 1.0
        
    def setInput(self, input_value):
        """Sets the input/output value for input nodes and """
        if self.node_type == Node.INPUT_NODE:
            self.inputValue = input_value
            self.outputValue = input_value
    
    
    
    def calculateOutput(self):
        """Computes output for the hidden or output layer nodes"""
        
        if (self.node_type == Node.HIDDEN or self.node_type == Node.OUTPUT):
            weightedSum = 0.0
            for parent in self.parents:
                weightedSum = weightedSum + parent.node.outputValue * parent.weight
        
            #compute output for hidden layer node
            if(self.node_type == Node.HIDDEN):
                self.outputValue = max(0, weightedSum) 
                
            #compute output for output layer node
            else:
                self.outputValue = math.exp(weightedSum)         
                
    
    def calculateSoftMax(self, total):
        """Computes softmax for the output layer nodes"""
        if self.node_type == Node.OUTPUT:
            self.outputValue = self.outputValue/total
    
    
    def calculateDeltaOutput(self, targetOutput):
        if self.node_type == Node.OUTPUT:
            self.delta = targetOutput - self.outputValue

    
    
    def calculateDeltaHidden(self, outputNodes):
        if self.node_type == Node.HIDDEN:
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
        if self.node_type == Node.HIDDEN or self.node_type == Node.OUTPUT:
            for nwp in self.parents:
                deltaWeight = learning_rate * nwp.node.outputValue * self.delta
                nwp.weight = nwp.weight + deltaWeight
    
class Instance():
    
    """
    Represents an instance object: a piece data to be predicted or trained on
    """
    value = None
    
    
    def __init__(self, attribute_list, class_value=None):
        self.attributes=attribute_list
        self.value = class_value
    
    def add_attr(self, value):
        self.attributes.append(value)
        
    def getValue(self):
        return self.class_value