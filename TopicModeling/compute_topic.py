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
# Import Logging
import logging

# Import local files
import get_data
from manage_path import *

def data_groupby():
    data = get_data.load_data(file_name="TRACE2014_jinming.pkl")
    data_gb_sell = data.groupby(by=['document_sell','BOND_SYM_ID'])
    data_gb_sell.astype(np.int16)
    data_gb_buy = data.groupby(by=['document_buy','BOND_SYM_ID'])
    data_gb_buy.astype(np.int16)
    return (data_gb_sell,data_gb_buy)

def compute_matrix1():
    """Compute matrix_1 which is count of bonds on given dealer and day"""
    data_gb_sell,data_gb_buy = data_groupby()
    print("computing matrix_1 ......")
    matrix_1 = data_gb_sell['BOND_SYM_ID'].size().unstack(fill_value=0)
    matrix_1 = matrix_1.append(data_gb_buy['BOND_SYM_ID'].size().unstack(fill_value=0))
    matrix_1 = matrix_1.sort_index(axis=1)
    print("computing matrix_1 done!")
    return matrix_1

def compute_matrix2():
    data_gb = data_groupby()
    print("computing matrix_2 ......")
    matrix_2 = data_gb['ENTRD_VOL_QT'].sum().unstack(fill_value=0)
    matrix_2 = matrix_2.sort_index(axis=1)
    print("computing matrix_2 done!")
    return matrix_2

def compute_matrix3():
    data_gb = data_groupby()
    print("computing matrix_3 ......")
    data_gb['cap'] = pd.eval(data_gb['ENTRD_VOL_QT'] * data_gb['RPTF_PR'])
    matrix_3 = data_gb['cap'].sum().unstack(fill_value=0)
    matrix_3 = matrix_3.sort_index(axis=1)
    print("computing matrix_3 done!")
    return matrix_3

def compute_corpus(matrix):
    """Compute corpus given a matrix"""
    corpus = gensim.matutils.Dense2Corpus(matrix.values,documents_columns=False)
    return corpus

def save_corpus(corpus,corpus_save_name):
    """Save the corpus given copus object and """
    corpus_directory = get_corpus_directory()
    if not corpus_directory.is_dir():
        create_directory(corpus_directory)
    file_name = corpus_directory / "{}.mm".format(corpus_save_name)
    gensim.corpora.MmCorpus.serialize(str(file_name), corpus)
    
def load_corpus(file_name):
    """Load the saved corpus"""
    print("loading corpus...")
    corpus_directory = get_corpus_directory()
    file_name = corpus_directory / "{}.mm".format(file_name)
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
    if(save):
        print("saving id2word ...")
        id2word_directory = get_id2word_directory()
        file_name = id2word_directory / "{}.npy".format(matrix_name)
        # save the id2word using numpy
        np.save(file_name, id2word)
        print("id2word saved!!")
    else:
        return id2word

def load_id2word(id2word_name):
    print("loading id2word ...")
    id2word_directory = get_id2word_directory()
    id2word_save_path = id2word_directory / "{}.npy".format(id2word_name)
    # load the id2word using numpy
    id2word = np.load(id2word_save_path).item()
    print("id2word loaded!!")
    return id2word

def compute_topic(corpus_name,corpus,num_topics,id2word,workers=3,chunksize=25000,passes=40,iterations=600):
    logs_directory = get_logs_directory()
    filename = "{}_{}topics.log".format(corpus_name,num_topics)
    log_filename = logs_directory / filename
    logging.basicConfig(filename=log_filename,format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("LdaMulticore Start!!")
    lda = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,id2word=id2word,workers=workers, num_topics=num_topics, chunksize=chunksize \
                                                  , passes=passes,iterations=iterations,dtype=np.float64,random_state=1)
    print("LdaMulticore Done!!")
    
    model_name = "{}_{}topics".format(corpus_name,num_topics)
    print("Saving Model as "+model_name)
    
    # create directory
    save_path = get_LDAModel_directory()
    # create sub-directory
    save_path = save_path / ("./{}/".format(model_name))
    create_directory(save_path)
    
    save_path = save_path / model_name
    save_path = datapath(str(save_path))

    lda.save(save_path)
    print("Model successfully save at" + save_path)
	
def compute_topic_distributed(corpus_name,corpus,num_topics,id2word,chunksize=25000,passes=40,iterations=600):
    logs_directory = get_logs_directory()
    filename = "{}_{}topics.log".format(corpus_name,num_topics)
    log_filename = logs_directory / filename
    logging.basicConfig(filename=log_filename,format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("LdaMulticore Start!!")
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word, num_topics=num_topics, chunksize=chunksize, passes=passes \
                                          ,iterations=iterations,distributed=True,dtype=np.float64,random_state=1)
    print("LdaMulticore Done!!")
    
    model_name = "{}_{}topics".format(corpus_name,num_topics)
    print("Saving Model as "+model_name)
    
    save_path = get_LDAModel_directory()
    # create sub-directory
    save_path = save_path / ("./{}/".format(model_name))
    create_directory(save_path)
    
    save_path = save_path / model_name
    save_path = datapath(str(save_path))

    lda.save(save_path)
    print("Model successfully save at" + save_path)

def main():
    corpus_name = str(input("Please enter corpus_name: "))
    num_topics = int(input("Please enter num_topics: "))
    workers = int(input("Please enter number of workers: "))
    passes = int(input("Please enter number of passes: "))
    if(corpus_name == 'matrix_1' or corpus_name == 'matrix1'):
        corpus = load_corpus("matrix_1")
        id2word = load_id2word("matrix_1")
    else:
        corpus = load_corpus("matrix_1")
        id2word = load_id2word("matrix_1")
    
    compute_topic(corpus_name,corpus,num_topics,id2word,workers=workers,passes=passes)
    
if __name__== "__main__":
    main()