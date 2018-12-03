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

def data_groupby():
    data = get_data.get_data()
    data_gb = data.groupby(by=['document','BOND_SYM_ID'])
    return data_gb

def get_matrix1():
    data_gb = data_groupby()
    print("computing matrix_1 ......")
    matrix_1 = data_gb['BOND_SYM_ID'].size().unstack(fill_value=0)
    matrix_1 = matrix_1.sort_index(axis=1)
    print("computing matrix_1 done!")
    return matrix_1

def get_id2word(matrix):
    le = preprocessing.LabelEncoder()
    le.fit(matrix.columns)
    transform = le.transform(matrix.columns)
    inverse_transform = le.inverse_transform(transform)
    id2word = dict(zip(transform, inverse_transform))
    return id2word

def compute_topic(matrix,matrix_name,num_topics,workers=3,chunksize=10000,passes=20,iterations=50):
    matrix_corpus = gensim.matutils.Dense2Corpus(matrix.values,documents_columns=False)
    id2word = get_id2word(matrix)
    print("LdaMulticore Start!!")
    lda = gensim.models.ldamulticore.LdaMulticore(corpus=matrix_corpus,id2word=id2word,workers=workers, num_topics=num_topics, chunksize=chunksize, passes=passes,iterations=iterations)
    print("LdaMulticore Done!!")
    
    model_name = "{}_{}topics".format(matrix_name,num_topics)
    print("Saving Model as "+model_name)
    
    current_path = os.getcwd()
    current_path = Path(current_path)
    save_path = current_path.parent / ("./Data/LDAModel/{}/".format(model_name))
    
    try:
        os.mkdir(save_path)
    except OSError:  
        print ("Creation of the directory %s failed" % save_path)
    else:  
        print ("Successfully created the directory %s " % save_path)
    save_path = save_path / model_name
    save_path = datapath(str(save_path))

    lda.save(save_path)
    print("Model successfully save at" + save_path)

def main():
    matrix_name = str(input("Please enter matrix_name"))
    num_topics = int(input("Please enter num_topics"))
    if(matrix_name == 'matrix_1' or matrix_name == 'matrix1'):
        matrix = get_matrix1()
    else:
        matrix = get_matrix1()
    id2word = get_id2word(matrix)
    compute_topic(matrix,matrix_name,num_topics)
    
if __name__== "__main__":
    main()