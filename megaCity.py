# -*- coding: utf-8 -*-
"""
megaCity.py

Usage:
  megaCity.py (-h | --help)
  megaCity.py --version
  megaCity.py word2vec DIRECTORY [-u MODEL] [-v]
  megaCity.py word2vec DIRECTORY [-o OUTPUTMODEL] [-v]
  megaCity.py tsneplot <model> <word> [-s <dest> <plotname>] [-v]
  megaCity.py wordVectorsList <inputmodel>

Arguments:
  <dest>                  destination folder for the tsne plot  
  <plotname>              name of the tsne plot to be saved
  <model>                 model to be used for creating a plot
  <word>                  word whose most similar words are to be plotted
  <inputmodel>            input model for displaying all word vectors

Options:
  -h --help               Show this screen.
  --version               Show version.
  -o OUTPUTMODEL          Specify the name of the output model
  -s <dest> <plotname>    Specify the destination folder and the name of the plot to be saved (for the tsne command)
  -u MODEL                Specify the name of the model to update
  -v                      Verbose output

"""


from docopt import docopt
import os
from sys import argv
from nltk.tokenize import TreebankWordTokenizer
import nltk.data
from gensim.models import word2vec
import logging
import time
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
from sklearn.manifold import TSNE

class Corpus(object):
    """Iterator for feeding sentences to word2vec"""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        
        #word_tokenizer = TreebankWordTokenizer()
        #sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        text = ''

        for root, dirs, files in os.walk(self.dirname):

            for file in files:

                if file.endswith(".txt"):
                    
                    file_path = os.path.join(root, file)


                    with open(file_path, 'r') as f:
                        for line in f:
                            line.decode('utf-8')
                            yield line.split()
                       
                        # text = f.read().decode('utf-8')
                        # sentences = sent_tokenizer.tokenize(text)
                       
                        # for sent in sentences:
                        #     yield word_tokenizer.tokenize(sent)
                       

def runWord2Vec(dir, modelName, verbose):

    """Runs the word2vec algorithm and saves the model under 'modelName' """
    
    if verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    rootdir = os.path.abspath(dir) 
    
    
    sentences = Corpus(rootdir)
   
    model = word2vec.Word2Vec(sentences, size=100, workers = 4)
    model.save(modelName)

def updateWord2Vec(dir, modelName, verbose):

    """Updates the word2vec model with additional txt files inside dir"""
    if verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    rootdir = os.path.abspath(dir) 
    

    loaded_model = word2vec.Word2Vec.load(modelName)
    more_sentences = Corpus(rootdir)
    loaded_model.train(more_sentences)
    loaded_model.save(modelName)

def wordVectorsList (modelName):

    """dumps all the word vectors on the screen; 
    these can be piped into a txt file via python megaCity.py wordVectorsList impgaze > file.txt
    """
    model = word2vec.Word2Vec.load(modelName)

    vocab = []
    for key, value in model.vocab.iteritems():
        vocab.append(key)
        
    sorted(vocab)

    # if verbose:
    #     "Printing all the word vectors inside the model " + modelName + "\n"

    for word in sorted(vocab):
        print "\n" + word + "\n" 
        print model[word]

def tsnePlot(plotname, modelName, word, dest):
    
    """Plots a tsne graph of words most similar to the word passed in the argument (as represented in the model previously calculated)"""
    
    model = word2vec.Word2Vec.load(modelName)
    words = [model.most_similar(word)[i][0] for i in range(0, len(model.most_similar(word)))]
    words.append(word)

    #nested list constaining 100 dimensional word vectors of each most-similar word
    
    word_vectors = [model[word] for word in words]
    word_vectors = np.array(word_vectors)

    tsne_model = TSNE(n_components=2, random_state=0)
    Y = tsne_model.fit_transform(word_vectors)
    sb.plt.plot(Y[:,0], Y[:,1], 'o') 

    for word, x, y in zip(words, Y[:,0], Y[:,1]):  
        sb.plt.annotate(word, (x, y), size=12)
        #sb.plt.pause(10)

    plotname = plotname + ".png"

    if not os.path.exists(dest):
        os.makedirs(dest)

    path = os.path.join(dest, plotname)

    sb.plt.savefig(path)

    

def main(arguments):

    
    modelName = "impgaze" #default model name

    if arguments['word2vec'] and arguments['DIRECTORY'] and arguments['-o'] == False and arguments['-u'] == False:

        dir = arguments['DIRECTORY']
        verbose = arguments['-v']
        runWord2Vec(dir, modelName, verbose)


    
    if arguments['word2vec'] and arguments['DIRECTORY'] and arguments['-o']:

        dir = arguments['DIRECTORY']
        verbose = arguments['-v']
        modelName = arguments['-o']
        runWord2Vec(dir, modelName, verbose)

    if arguments['word2vec'] and arguments['DIRECTORY'] and arguments['-u']:

        dir = arguments['DIRECTORY']
        modelName = arguments['-u'] #existing model for update
        verbose = arguments['-v']
        updateWord2Vec(dir, modelName, verbose)

    if arguments['tsneplot'] and arguments['<model>'] and arguments['<word>'] and arguments['-s'] and arguments['<plotname>']:
        
        modelName = arguments['<model>']
        word = arguments['<word>']
        dest = arguments['-s']
        plotname = arguments['<plotname>']
        verbose = arguments['-v']

        tsnePlot(plotname, modelName, word, dest)

        if verbose:
            print "Plot " + plotname + " has been saved inside the folder " + dest 

    if arguments['wordVectorsList'] and arguments['<inputmodel>']:
        
        modelName = arguments['<inputmodel>']
        wordVectorsList(modelName, verbose)

      

if __name__ == '__main__':

    arguments = docopt(__doc__, version='megaCityGazing 1.0')
    main (arguments)
