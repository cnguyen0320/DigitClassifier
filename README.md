# DigitClassifier

Training data from Semeion Handwritten Digit Data Set:
https://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit

## Training
Training is completed by running train.py.
Maximum success rate is found on a tuning set and the layer weights are saved into files named hiddenweights.txt and outputweights.txt

`python train.py <parameter_file.json>`
###Options:

  -d prints out statistics on loss and tuning set success rate on each epoch
  
  -s prevents overwriting the saved hiddenweights.txt and outputweights.txt files

## Prediction
Prediction is completed by running predict.py

To run with an input image file:

`python train.py <parameter_file> <hiddenweights_file> <outputweights_file>` <inputfile>'

To run with a hand drawn input using the mouse and an on-screen GUI:

`python train.py <parameter_file> <hiddenweights_file> <outputweights_file>` -d'

Press <esc> or <enter> during drawing to run the prediction
Press <r> to reset the drawing
