# BigramHMM
Bigram HMM for part of speech tagging

To run bigramHMM tagger, use command:

python bigramHMM.py <training file> <test file> <output file>
ensure <output file> doesn't exist because it will be appended to for accuracy comparison with <test file>

example:
python bigramHMM.py twt.train.json twt.test.json testout.json

I commented out the confusion matrix printing in case the library package doesn't exist on the machine it is run on
