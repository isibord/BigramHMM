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
      printline = 0
      for j in range(len(currExpected)):
        totalcount += 1
        expectedArrary.append(currExpected[j][1])
        actualArray.append(currActual[j][1])
        if currExpected[j] == currActual[j]:
          totalmatch += 1
        else:
          if printline == 0:
            #print 'exp: ' + str(currExpected)
            #print 'actual: ' + str(currActual)
            printline = 1

    """
    y_pred = pd.Series(actualArray, name='Predicted')
    y_exp = pd.Series(expectedArrary, name='Actual')
    df_confusion = pd.crosstab(y_exp, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    print df_confusion
    """

    print("accuracy %d/%d: %.2f%%" % (totalmatch, totalcount, 100*(totalmatch/totalcount)))


class TrainTrigramHMM:
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
    self.trigramToCount = defaultdict(lambda: 0)

    self.trainFromFile()

  def trainFromFile(self):
    with open(self.filepath, 'r') as f:
      for line in f:
        self.sentenceList.append(json.loads(line))

    #each sentence training
    for sentence in self.sentenceList:
      prevTag1 = TrainTrigramHMM.startState
      prevTag2 = TrainTrigramHMM.startState
      self.tagToCount[prevTag1] += 1
      self.tagToCount[prevTag2] += 1
      for word in sentence:
        currWord = word[0].encode('utf-8')
        currTag = word[1].encode('utf-8')

        self.tagToCount[currTag] += 1
        self.trigramToCount[(prevTag1, prevTag2, currTag)] += 1
        self.emissToCount[(currWord, currTag)] += 1
        self.transToCount[(prevTag1, prevTag2)] += 1

        prevTag1 = prevTag2
        prevTag2 = currTag

      #For last word to stop symbol
      currWord = TrainTrigramHMM.stopSymbol
      currTag = TrainTrigramHMM.stopState

      self.tagToCount[currTag] += 1
      self.trigramToCount[(prevTag1, prevTag2, currTag)] += 1
      self.emissToCount[(currWord, currTag)] += 1
      self.transToCount[(prevTag1, prevTag2)] += 1

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
    prevT1, prevT2, currT = transData
    countTrans = self.transToCount[(prevT1, prevT2)]
    if countTrans == 0:
      countTrans = 0.0000002 #if transition wasn't seen in test
    countTri = self.trigramToCount[transData]
    # add k smoothing
    score = log(countTri + 0.2) - (log(countTrans) + (0.2 * len(self.tagToCount)))
    return score


class TagSentencesInFile:

  def __init__(self, filepath, trigramHMM, outFile="outfile.json"):
    self.filepath = filepath
    self.sentenceList = []
    self.outputFile = outFile
    self.trigramHMM = trigramHMM
    self.tagList = trigramHMM.tagToCount.keys()
    self.tagList.insert(0, self.tagList.pop(self.tagList.index(TrainTrigramHMM.startState)))
    self.loadFileData()

  def loadFileData(self):
    with open(self.filepath, 'r') as f:
      for line in f:
        self.sentenceList.append(json.loads(line))

  def tagEachSentence(self, sentence):
    wordCount = len(sentence)

    ########### Initialize matrices #################
    Viterbi = {}
    finalpath = {}

    Viterbi[0, TrainTrigramHMM.startState, TrainTrigramHMM.startState] = 0.0
    finalpath[TrainTrigramHMM.startState, TrainTrigramHMM.startState] = []

    ########### Start Viterbi #######################
    for t in xrange(1, wordCount):
      temppath = {}
      currObs = sentence[t][0]
      for tag in self.returnTagList(t-1):
        for tag1 in self.returnTagList(t):
          Viterbi[t, tag, tag1], past = max(
            [(Viterbi[t - 1, currTag, tag] + self.trigramHMM.getTransProb((currTag, tag, tag1)) + self.trigramHMM.getEmissProb((currObs, tag1)), currTag) for currTag in self.returnTagList(t-2)])
          temppath[tag, tag1] = finalpath[past, tag] + [tag1]
      finalpath = temppath

    ###### finish viterbi with final state ######

    ###backtrack!
    pval, tagmax, tag1max = max([(Viterbi[wordCount-1, tag, tag1] + self.trigramHMM.getTransProb((tag, tag1, TrainTrigramHMM.startState)), tag, tag1) for tag in self.tagList for tag1 in self.tagList])

    returnpath = finalpath[tagmax, tag1max]
    returnpath.insert(0, TrainTrigramHMM.startState)
    return returnpath

  #only return start state if in first iteration of loop
  def returnTagList(self, count):
    if count == -1 or count == 0:
      return [TrainTrigramHMM.startState]
    else:
      return self.tagList

  def tagSentences(self):
    counting = 0
    for sentence in self.sentenceList:
      counting += 1
      print 'tagging sentence # %d' % counting
      sentence.insert(0, [TrainTrigramHMM.startSymbol, TrainTrigramHMM.startState])
      sentence.append([TrainTrigramHMM.stopSymbol, TrainTrigramHMM.stopState])
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
  trigramHMM = TrainTrigramHMM(trainPath)
  print 'generated trigram HMMs \n'

  tagger = TagSentencesInFile(testPath, trigramHMM, outpath)
  print 'tagger created \n'
  tagger.tagSentences()

  print 'sentences tagged \n measuring accuracy \n'

  accuracy = MeasureAccuracy(testPath, outpath)


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2], sys.argv[3])