# DigitClassifier

Training data from Semeion Handwritten Digit Data Set:
https://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit

##Training
Training is completed by running train.py
`python train.py <parameter_file.json>`

Maximum success rate is found on a tuning set and the layer weights are saved into files named hiddenweights.txt and outputweights.txt

##Prediction
Prediction is completed by running predict.py
`python train.py <parameter_file> <hiddenweights_file> <outputweights_file>`

