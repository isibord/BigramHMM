from __future__ import division
from collections import defaultdict
import sys, os
import json
import math
import string
from math import log, pow
#import pandas as pd

class MeasureAccuracy:

  """
  Measure the accuracy of the output of tagged tweets with respect to the expected tags.
  Expected and actual both have to have the exact same tweets for this to work
  """
  def __init__(self, expected, predicted):
      self.expectedFilePath = expected
      self.actualFilePath = predicted
      self.expectedLines = []
      self.actualLines = []
      if self.readfiles():
        self.calculateAccuracy()

  """
  Load the data from expected and actual files
  """
  def readfiles(self):
    with open(self.expectedFilePath, 'r') as f:
      for line in f:
        self.expectedLines.append(json.loads(line))

    with open(self.actualFilePath, 'r') as f:
      for line in f:
        self.actualLines.append(json.loads(line))

    if len(self.expectedLines) != len(self.actualLines):
      print('ERROR: Expected and actual file lengths dont match')
      return False

    return True

  """
  Go through each tweet and measure the % accuracy
  """
  def calculateAccuracy(self):
    totalcount = 0
    totalmatch = 0
    expectedArrary = []
    actualArray = []
    for i in range(len(self.expectedLines)):
      currExpected = self.expectedLines[i]
      currActual = self.actualLines[i]
      for j in range(len(currExpected)):
        totalcount += 1
        expectedArrary.append(currExpected[j][1])
        actualArray.append(currActual[j][1])
        if currExpected[j] == currActual[j]:
          totalmatch += 1

    """
    y_pred = pd.Series(actualArray, name='Predicted')
    y_exp = pd.Series(expectedArrary, name='Actual')
    df_confusion = pd.crosstab(y_exp, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    print df_confusion
    """

    print("accuracy %d/%d: %.2f%%" % (totalmatch, totalcount, 100*(totalmatch/totalcount)))

class TrainBigramHMM:
  startSymbol = '<s>'
  stopSymbol = '</s>'
  startState = '<s>'
  stopState = '</s>'

  def __init__(self, filepath):
    self.filepath = filepath
    self.sentenceList = []
    self.tagToCount = defaultdict(lambda: 0)
    self.transToCount = defaultdict(lambda: 0)
    self.emissToCount = defaultdict(lambda: 0)

    self.trainFromFile()

  def trainFromFile(self):
    with open(self.filepath, 'r') as f:
      for line in f:
        self.sentenceList.append(json.loads(line))

    #each sentence training
    for sentence in self.sentenceList:
      prevWord = TrainBigramHMM.startSymbol
      prevTag = TrainBigramHMM.startState
      self.tagToCount[prevTag] += 1
      for word in sentence:
        currWord = word[0].encode('utf-8')
        currTag = word[1].encode('utf-8')

        self.tagToCount[currTag] += 1
        self.transToCount[(prevTag, currTag)] += 1
        self.emissToCount[(currWord, currTag)] += 1

        prevWord = currWord
        prevTag = currTag

      #For last word to stop symbol
      currWord = TrainBigramHMM.stopSymbol
      currTag = TrainBigramHMM.stopState

      self.tagToCount[currTag] += 1
      self.transToCount[(prevTag, currTag)] += 1
      self.emissToCount[(currWord, currTag)] += 1

  def getEmissProb(self, emissData):
    word, tag = emissData
    count = self.emissToCount[emissData]
    if count == 0: #check for OOV  UNK-EMOTICON, UNK-URL, UNK-MENTION, UNK-HASHTAG
      if word.startswith('\u') and tag == 'E':
        return 0.0 #log prob of 1
      elif word.startswith('http') and tag == 'U':
        return 0.0
      elif word.startswith('@') and tag == '@':
        return 0.0
      elif word.startswith('#') and tag == '#':
        return 0.0
    #add k smoothing
    score = log(count + 0.2) - (log(self.tagToCount[tag] + (0.2 * len(self.tagToCount))))
    return score

  def getTransProb(self, transData):
    prevT, currT = transData
    count = self.transToCount[transData]
    # add k smoothing
    score = log(count + 0.2) - (log(self.tagToCount[prevT]) + (0.2 * len(self.tagToCount)))
    return score


class TagSentencesInFile:

  def __init__(self, filepath, bigramHMM, outFile="outfile.json"):
    self.filepath = filepath
    self.sentenceList = []
    self.outputFile = outFile
    self.bigramHMM = bigramHMM
    self.tagList = bigramHMM.tagToCount.keys()
    self.neg_infinity = float("-infinity")
    self.tagList.insert(0, self.tagList.pop(self.tagList.index(TrainBigramHMM.startState)))
    self.loadFileData()

  def loadFileData(self):
    with open(self.filepath, 'r') as f:
      for line in f:
        self.sentenceList.append(json.loads(line))

  def tagEachSentence(self, sentence):
    wordCount = len(sentence)
    #print 'Len Sentence: %s' % wordCount
    scoreMatrix = []
    # scoreMatrix is the table the Viterbi algorithm updates
    # it is an array of length wordCount + 1
    # each element will be an array of length len(tagList)
    backTrack = []
    # backTrack is the best scoring tag to transition from get to curr state

    ########### Initialize matrices #################
    for i in xrange(wordCount):
      scores = defaultdict(lambda: self.neg_infinity)
      states = {}
      for tag in self.tagList:
        if i==0 and tag == TrainBigramHMM.startState: #initial states
          scores[tag] = 0.0 #log prob of 0 is prob of 1, start state always has that
          states[tag] = TrainBigramHMM.startState
        else:
          scores[tag] = self.neg_infinity # log prob of neg inf, is prob of 0
          states[tag] = 'init' #placeholder
      scoreMatrix.append(scores)
      backTrack.append(states)

    ########### End initialize ######################

    ########### Start Viterbi #######################

    for t in xrange(1, wordCount):
      currObs = sentence[t][0]
      #print 'Word: %s' % currObs
      #print 'Idx: %s' % t

      for tag in self.tagList: #curr tag
        for tag1 in self.tagList: #prev tag
            #print '  ptag,  tag: %s ,  %s' % (tag1, tag)
            #prev score + transition prob + emission prob
            score = scoreMatrix[t-1][tag1] + self.bigramHMM.getTransProb((tag1, tag)) + self.bigramHMM.getEmissProb((currObs, tag))
            if score > scoreMatrix[t][tag]:
              scoreMatrix[t][tag] = score
              backTrack[t][tag] = tag1
              #print 'MAX  ptag,  tag: %s ,  %s' % (tag1, tag)

    ###### finish viterbi with final state ######
    bestTag = 'sometag' #dummy tag
    for tag in self.tagList:
      if scoreMatrix[wordCount-1][tag] > scoreMatrix[wordCount-1][bestTag]:
        bestTag=tag

    path = [bestTag]
    for t in xrange(wordCount-1, 0, -1): #count backwards
      bestTag = backTrack[t][bestTag]
      path[0:0] = [bestTag]

    return path

  def tagSentences(self):
    for sentence in self.sentenceList:
      sentence.insert(0, [TrainBigramHMM.startSymbol, TrainBigramHMM.startState])
      sentence.append([TrainBigramHMM.stopSymbol, TrainBigramHMM.stopState])
      path = self.tagEachSentence(sentence)
      self.writeSentenceOutput(sentence, path)

  def writeSentenceOutput(self, sentence, path):
    finalout = []
    with open(self.outputFile, 'a+') as f:
      for i in xrange(1, len(sentence)-1):
        finalout.append([sentence[i][0], path[i]])
      json.dump(finalout, f)
      f.write('\n')

def main(trainPath='twt.train.json', testPath='twt.test.json', outpath='testout.json'):

  bigramHMM = TrainBigramHMM(trainPath)
  print 'generated bigram HMMs \n'

  tagger = TagSentencesInFile(testPath, bigramHMM, outpath)
  print 'tagger created \n'
  tagger.tagSentences()

  print 'sentences tagged \n measuring accuracy \n'

  accuracy = MeasureAccuracy(testPath, outpath)


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2], sys.argv[3])