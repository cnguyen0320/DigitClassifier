#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import NN
import json, sys, cv2
import numpy as np
from upload_img import resize2binary, crop2fit

class Sketcher:
        def __init__(self, windowname, dests, colors_func, image_size):
            self.prev_pt = None
            self.windowname = windowname
            self.dests = dests
            self.colors_func = colors_func
            self.dirty = False
            self.show()
            self.size = image_size
            cv2.setMouseCallback(self.windowname, self.on_mouse)
    
        def show(self):
            cv2.imshow(self.windowname, self.dests[0])
    
        def on_mouse(self, event, x, y, flags, param):
            pt = (x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.prev_pt = pt
            elif event == cv2.EVENT_LBUTTONUP:
                self.prev_pt = None
    
            if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
                for dst, color in zip(self.dests, self.colors_func()):
                    cv2.line(dst, self.prev_pt, pt, color, int(self.size/15))
                self.dirty = True
                self.prev_pt = pt
                self.show()

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
    
    numAttributes = parameters['data']['num_attributes']
    numDigits = parameters['data']['num_classes']
    numHiddenNodes = parameters['network']['num_hidden_nodes']
    seed = parameters['network']['seed']
    
    return numAttributes,numDigits,numHiddenNodes, seed


#def test(network, trainingSet):
#    count = 0
#    for inst in trainingSet:
#        if inst.value == network.predict(inst):
#            count = count + 1
#    return count/len(trainingSet)

def draw2instance():
    size = 500
    file_name = 'drawing.jpg'

    blank = np.full([size,size],255, np.uint8) # creates a 2D array of all 0s to create a black canvas
    sketch = Sketcher('drawing', [blank, blank], lambda : ((255, 255, 255), 0), size) # create blank canvas
    print('Press <esc> or <enter> to finish. <r> to reset image.')
    
    while True:
        ch = cv2.waitKey()
        if ch == 27 or ch == 10 or ch == 13: #esc or enter key finishes
            break
        if ch == ord('r'): #r key refreshes the image
            blank[:] = blank
            blank[:] = 255
            sketch.show()
            
    cv2.destroyAllWindows()
    
    
    im_bw = cv2.threshold(sketch.dests[0], 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    im_bw = crop2fit(im_bw) #crop the image it to fit
    
    width = 16
    height = 16
    
    image = cv2.resize(im_bw, (height,width),interpolation = cv2.INTER_AREA)
        
    flatten = []
    for i in range(width):
            for j in range(height): 
                flatten.append(int(not bool(image[i][j])))
    
    instance = NN.Instance(flatten, None)
    return instance

if __name__ == '__main__':
    
    try:
        #get filenames from the command line argument and load them
        json_file = str(sys.argv[1])
        hiddenfile = str(sys.argv[2])
        outputfile = str(sys.argv[3])
        inputimage = str(sys.argv[4])
        if '-d' in inputimage:
            instance = draw2instance()
        else:
            instance = NN.Instance(resize2binary(inputimage))
            
        hiddenWeights = loadWeights(hiddenfile)
        outputWeights = loadWeights(outputfile)
        numAttributes, numDigits,numHiddenNodes,seed = loadJSON(json_file)
        
    except:
        print('Error loading. usage: python train.py <parameter_file> <hiddenweights_file> <outputweights_file> <input_image>' )
        print('Enter -d for <input image> to draw an input image')
        #exit(1)
    
    
    #Creates a network using the parameters set in train.py and the loaded hidden and output weights
    network = NN.Network(None, None, numAttributes, numDigits, numHiddenNodes, None, None, hiddenWeights, outputWeights)
    
    prediction, confidence = network.predict(instance)
    
    print('Prediction:',prediction, "with", int(confidence*100), "percent confidence")

#    inst_list = train.parseDataSet(training_file, numAttributes)    
#    random_index = math.floor(random.random()*len(inst_list))    
#    inst = inst_list[random_index]
#    
#    print("Known value:", inst.value)
#    
#    print("The detected digit is a", network.predict(inst))
#    
#    print("success rate: ", test(network, inst_list))