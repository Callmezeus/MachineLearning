# svm.py
# -------------
# svm implementation
import util
import numpy as numpyth
from sklearn.svm import *
PRINT = True

class SVMClassifier:
  """
  svm classifier
  """
  def __init__( self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "svm"
    self.clf = LinearSVC(multi_class='ovr')
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    print "Starting SVM ..."
    trainArray = []
    for i in range(len(trainingData)):
      trainArray.append(trainingData[i].values()) 
    datum = numpyth.array(trainArray) 
    labels = numpyth.array(trainingLabels)
    self.clf.fit(datum, labels)
    
  def classify(self, data ):
    counter=0
    guessArray = []
    for datum in data:
      datumArray = numpyth.array([datum.values()])
      guess = self.clf.predict(datumArray) 
      guessArray.append(guess)
    return guessArray