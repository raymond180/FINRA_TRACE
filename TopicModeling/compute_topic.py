# Import Library
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn import preprocessing
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.test.utils import datapath
# Import local files
import get_data
from manage_path import get_current_directory,create_directory

def data_groupby():
    data = get_data.get_data()
    data_gb = data.groupby(by=['document','BOND_SYM_ID'])
    return data_gb

def compute_matrix1():
    data_gb = data_groupby()
    print("computing matrix_1 ......")
    matrix_1 = data_gb['BOND_SYM_ID'].size().unstack(fill_value=0)
    matrix_1 = matrix_1.sort_index(axis=1)
    print("computing matrix_1 done!")
    return matrix_1

def compute_corpus(matrix):
    corpus = gensim.matutils.Dense2Corpus(matrix.values,documents_columns=False)
    return corpus

def save_corpus(corpus,file_name):
    current_path = os.getcwd()
    current_path = Path(current_path)
    corpus_save_path = current_path.parent / "./Data/Corpus/"
    try:
        os.mkdir(corpus_save_path)
    except OSError:  
        print ("Creation of the directory %s failed" % corpus_save_path)
    else:  
        print ("Successfully created the directory %s " % corpus_save_path)
    file_name = corpus_save_path / "{}.mm".format(file_name)
    gensim.corpora.MmCorpus.serialize(str(file_name), corpus)
    
def load_corpus(file_name):
    print("loading corpus...")
    current_path = os.getcwd()
    current_path = Path(current_path)
    corpus_load_path = current_path.parent / "./Data/Corpus/"
    file_name = corpus_load_path / "{}.mm".format(file_name)
    file_name = str(file_name)
    corpus = gensim.corpora.MmCorpus(file_name)
    print("corpus successfully loaded!!")
    print(corpus)
    return corpus

def compute_id2word(matrix,matrix_name,save=True):
    le = preprocessing.LabelEncoder()
    le.fit(matrix.columns)
    transform = le.transform(matrix.columns)
    inverse_transform = le.inverse_transform(transform)
    id2word = dict(zip(transform, inverse_transform))
    print("saving id2word ...")
    if(save):
        current_path = get_current_directory()
        id2word_save_path = current_path.parent / "./Data/id2word/"
        create_directory(id2word_save_path)
        file_name = id2word_save_path / "{}.npy".format(file_name)
        # save the id2word using numpy
        np.save(file_name, id2word)
        print("id2word saved!!")
    else:
        return id2word

def load_id2word(id2word_name):
    print("loading id2word ...")
    current_path = get_current_directory()
    id2word_save_path = current_path.parent / "./Data/id2word/"
    id2word_save_path = id2word_save_path / "{}.npy".format(id2word_name)
    # load the id2word using numpy
    id2word = np.load(id2word_save_path).item()
    print("id2word loaded!!")
    return id2word

def compute_topic(corpus_name,corpus,num_topics,id2word,workers=3,chunksize=10000,passes=20,iterations=50):
    print("LdaMulticore Start!!")
    lda = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,id2word=id2word,workers=workers, num_topics=num_topics, chunksize=chunksize, passes=passes,iterations=iterations)
    print("LdaMulticore Done!!")
    
    model_name = "{}_{}topics".format(corpus_name,num_topics)
    print("Saving Model as "+model_name)
    
    current_path = get_current_directory()
    save_path = current_path.parent / ("./LDAModel/{}/".format(model_name))
    # create directory
    create_directory(save_path)
    
    save_path = save_path / model_name
    save_path = datapath(str(save_path))

    lda.save(save_path)
    print("Model successfully save at" + save_path)

def main():
    corpus_name = str(input("Please enter corpus_name: "))
    num_topics = int(input("Please enter num_topics: "))
    workers = int(input("Please enter number of workers: "))
    if(corpus_name == 'matrix_1' or corpus_name == 'matrix1'):
        corpus = load_corpus("matrix_1")
        id2word = load_id2word("matrix_1")
    else:
        corpus = load_corpus("matrix_1")
        id2word = load_id2word("matrix_1")
    
    compute_topic(corpus_name,corpus,num_topics,id2word,workers=workers)
    
if __name__== "__main__":
    main()