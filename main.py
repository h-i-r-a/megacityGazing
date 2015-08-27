import os
from sys import argv
from nltk.tokenize import TreebankWordTokenizer
import nltk.data
from gensim.models import word2vec
import logging
import time
import seaborn as sb
from sklearn.manifold import TSNE
import numpy as np


class Corpus(object):
    """Iterator for feeding sentences to word2vec"""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        
        word_tokenizer = TreebankWordTokenizer()
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        text = ''

        for root, dirs, files in os.walk(self.dirname):

            for file in files:

                if file.endswith(".txt"):
                    
                    file_path = os.path.join(root, file)


                    with open(file_path, 'r') as f:
                       
                        text = f.read().decode('utf-8')
                        sentences = sent_tokenizer.tokenize(text)
                       
                        for sent in sentences:
                            yield word_tokenizer.tokenize(sent)
                       
def tsnePlot(word, model):
  
    """Plots a tsne graph of words most similar to the word passed in the argument (as represented in the model previously calculated)"""
    #arguments: word + model ; mention in the --help
    
    words = [model.most_similar(word)[i][0] for i in range(0, len(model.most_similar(word)))]
    word.append(word)

    #nested list constaining 100 dimensional word vectors of each most-similar word
    
    word_vectors = [model[word] for word in words]
    word_vectors = np.array(word_vectors)

    tsne_model = TSNE(n_components=2, random_state=0)
    Y = tsne_model.fit_transform(word_vectors)
    sb.plt.plot(Y[:,0], Y[:,1], 'o') 

   

    for word, x, y in zip(words, Y[:,0], Y[:,1]):  
        sb.plt.annotate(word, (x, y), size=12)
        sb.plt.pause(10)
 



def vocab(model):
    vocab = []
    for key, value in model.vocab.iteritems():
        vocab.append(key)
    
    return vocab        
    

def main():
    """Runs gensim; creates a word2vec model, saves it as a binary file named 'impgaze'"""   
    rootdir = argv[1] 
    rootdir = os.path.abspath(rootdir) 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = Corpus(rootdir)
   
    model = word2vec.Word2Vec(sentences, size=100, workers = 4)
    model.save('impgaze')
 
    

if __name__ == "__main__":
    
   
    main()
    
