# mlp.py
# -------------

# mlp implementation
import util
from random import random
import numpy as n
PRINT = True

class MLPClassifier:
  """
  mlp classifier
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mlp"
    self.max_iterations = max_iterations
    self.network = []

  def train( self, trainingData, trainingLabels, validationData, validationLabels):
    ins = len(trainingData[0])
    outs = len(self.legalLabels)
    hiddens = 28
    network = list()

    hidden = [{'weights': [random() for i in range(ins + 1)]} for i in range(hiddens)]
    network.append(hidden)
    output_layer = [{'weights': [random() for i in range(hiddens + 1)]} for i in range(outs)]
    network.append(output_layer)
    self.network = network

    guesses = []
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      
      for i in range(len(trainingData)):
        trainingSet = list()
        trainingSet.extend(trainingData[i].values())
        trainingSet.append(trainingLabels[i])

        learnRate = 0.5
        totalError = 0
        outputs = trainingSet
        
        for layer in network:
          currInputs = []
          for percept in layer:
            thresh = percept['weights'][-1]
            for i in range(len(percept['weights']) - 1):
              thresh += percept['weights'][i] * outputs[i]
            percept['output'] = n.tanh(thresh / 100)
            currInputs.append(percept['output'])
          outputs = currInputs
        expected = [0 for i in range(outs)]
        expected[trainingSet[-1]] = 1
        totalError += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])

        for i in reversed(range(len(network))):
          layer = network[i]
          errors = list()
          if i != len(network) - 1:
            for j in range(len(layer)):
              error = 0.0
              for percept in network[i + 1]:
                error += (percept['weights'][j] * percept['delta'])
              errors.append(error)
          else:
            for j in range(len(layer)):
              percept = layer[j]
              errors.append(expected[j] - percept['output'])
          for j in range(len(layer)):
            percept = layer[j]
            percept['delta'] = errors[j] * (1 - percept['output']**2)

        for i in range(len(network)):
          inputs = trainingSet[:-1]
          if i != 0:
            inputs = [percept['output'] for percept in network[i - 1]]
          for percept in network[i]:
            for j in range(len(inputs)):
              percept['weights'][j] += learnRate * percept['delta'] * inputs[j]
            percept['weights'][-1] += learnRate * percept['delta']

        inputs = trainingSet
        
        for layer in network:
          currInputs = []
          for percept in layer:
            thresh = percept['weights'][-1]
            for i in range(len(percept['weights']) - 1):
              thresh += percept['weights'][i] * inputs[i]
            percept['output'] = n.tanh(thresh / 100)
            currInputs.append(percept['output'])
          inputs = currInputs
        
        prediction = inputs.index(max(inputs))
        guesses.append(prediction)

      correct=0
      
      for i in range(len(trainingLabels)):
        if guesses[i] == trainingLabels[i]:
          correct += 1

  
  def classify(self, data ):
    guesses = []
    for datum in data:
      # fill predictions in the guesses list
      "*** YOUR CODE HERE ***"
      trainingSet = list()
      trainingSet.extend(datum.values())
      guessLabel = random()
      trainingSet.append(guessLabel)
      inputs = trainingSet
      
      for layer in self.network:
        currInputs = []
        for percept in layer:
          thresh = percept['weights'][-1]
          for i in range(len(percept['weights']) - 1):
            thresh += percept['weights'][i] * inputs[i]
          percept['output'] = n.tanh(thresh / 100)
          currInputs.append(percept['output'])
        inputs = currInputs
      
      prediction = inputs.index(max(inputs))
      guesses.append(prediction)
    
    return guesses