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
from get_data import load_data
from manage_path import *

def create_document(first,second):
    return str(first) + ',' + str(second)

def create_document(first,second,third):
    return str(first) + ',' + str(second) + ',' + str(third)

def document_date2year(date):
    return str(date[0:4])

def compute_Dc_v1(data):
    """Compute Dc_v1 which is count of bonds on given dealer and day"""
    create_document_vectorize = np.vectorize(create_document)
    print("creating documents ......")
    # Add new column Dc_v1_S which is the string representation of report dealer buy on the specific day
    data['Dc_v1_S'] = create_document_vectorize(data['Report_Dealer_Index'].values , data['document_date'].values)
    # Add new column Dc_v1_B which is the string representation of report dealer sell on the specific day
    data['Dc_v1_B'] = create_document_vectorize(data['Contra_Party_Index'].values , data['document_date'].values)
    print("documents created!!")
    
    data_gb_sell = data.groupby(by=['Dc_v1_S','BOND_SYM_ID'])
    data_gb_buy = data.groupby(by=['Dc_v1_B','BOND_SYM_ID'])
    
    print("computing Dc_v1 ......")
    Dc_v1 = data_gb_sell['BOND_SYM_ID'].size().astype(np.int16).unstack(fill_value=0)
    Dc_v1 = Dc_v1.append(data_gb_buy['BOND_SYM_ID'].size().astype(np.int16).unstack(fill_value=0))
    # Sort index before groupby index
    Dc_v1 = Dc_v1.sort_index()
    # Groupby index and sum them to get the count of bonds for the dealer on the certain day
    Dc_v1 = Dc_v1.groupby(by=Dc_v1.index).sum()
    # Sort columns so we get nice format
    Dc_v1 = Dc_v1.sort_index(axis=1)
    print("computing Dc_v1 done!")
    return Dc_v1

def compute_Dc_v2(data):
    """Compute Dc_v2 which is count of bonds on given dealer and day seperated buy and sell"""
    create_document_vectorize = np.vectorize(create_document)
    print("creating documents ......")
    # Add new column Dc_v2_S which is the string representation of report dealer buy on the specific day
    data['Dc_v2_S'] = create_document_vectorize(data['Report_Dealer_Index'].values , data['document_date'].values , 'S')
    # Add new column Dc_v2_B which is the string representation of report dealer sell on the specific day
    data['Dc_v2_B'] = create_document_vectorize(data['Contra_Party_Index'].values , data['document_date'].values , 'B')
    print("documents created!!")
    
    data_gb_sell = data.groupby(by=['Dc_v2_S','BOND_SYM_ID'])
    data_gb_buy = data.groupby(by=['Dc_v2_B','BOND_SYM_ID'])
    
    print("computing Dc_v2 ......")
    Dc_v2 = data_gb_sell['BOND_SYM_ID'].size().astype(np.int16).unstack(fill_value=0)
    Dc_v2 = Dc_v2.append(data_gb_buy['BOND_SYM_ID'].size().astype(np.int16).unstack(fill_value=0))
    Dc_v2 = Dc_v2.sort_index(axis=1)
    print("computing Dc_v2 done!")
    return Dc_v2

def compute_Dc_v3(data):
    """Compute Dc_v2 which is count of bonds on given dealer and day seperated buy and sell"""
    create_document_vectorize = np.vectorize(create_document)
    print("creating documents ......")
    # Ignore Report_Dealer_Index that is '0' and Contra_Party_Index that is '99999'
    data = data.loc[(data['Report_Dealer_Index'] != '0') & (data['Contra_Party_Index'] != '99999')].copy()
    # Add new column Dc_v3 which is the string representation of report dealer buy on the specific day
    data['Dc_v3_S'] = create_document_vectorize(data['Report_Dealer_Index'].values , data['document_date'].values , 'S')
    # Add new column Dc_v3 which is the string representation of report dealer sell on the specific day
    data['Dc_v3_B'] = create_document_vectorize(data['Contra_Party_Index'].values , data['document_date'].values , 'B')
    print("documents created!!")
    
    data_gb_sell = data.groupby(by=['Dc_v3_S','BOND_SYM_ID'])
    data_gb_buy = data.groupby(by=['Dc_v3_B','BOND_SYM_ID'])
    
    print("computing Dc_v3 ......")
    Dc_v3 = data_gb_sell['BOND_SYM_ID'].size().astype(np.int16).unstack(fill_value=0)
    Dc_v3 = Dc_v3.append(data_gb_buy['BOND_SYM_ID'].size().astype(np.int16).unstack(fill_value=0))
    Dc_v3 = Dc_v3.sort_index(axis=1)
    print("computing Dc_v3 done!")
    return Dc_v3

def compute_Tc_v1(data):
    """Compute Tc_v1 which is a document will represent the triple (seller, bond, buyer, date) directly"""
    create_document_vectorize = np.vectorize(create_document)
    document_date2year_vectorize = np.vectorize(document_date2year)
    print("creating documents ......")
    # Add new column Dc_v3 which is the string representation of report dealer buy on the specific day
    data['document_date'] = document_date2year_vectorize(data['document_date'].values)
    data['Tc_v1_S_B_D'] = create_document_vectorize(data['Report_Dealer_Index'].values , data['Contra_Party_Index'].values , data['document_date'].values)
    print("documents created!!")
    
    data_gb = data.groupby(by=['Tc_v1_S_B_D','BOND_SYM_ID'])
    
    print("computing Tc_v1 ......")
    Tc_v1 = data_gb['BOND_SYM_ID'].size().astype(np.int16).unstack(fill_value=0)
    Tc_v1 = Tc_v1.sort_index(axis=1)
    print("computing Tc_v1 done!")
    return Tc_v1

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

def compute_corpus(matrix,corpus_save_name,save=True):
    """Compute corpus given a matrix and save it"""
    corpus = gensim.matutils.Dense2Corpus(matrix.values,documents_columns=False)
    if save:
        corpus_directory = get_corpus_directory()
        if not corpus_directory.is_dir():
            create_directory(corpus_directory)
        file_name = corpus_directory / "{}.mm".format(corpus_save_name)
        gensim.corpora.MmCorpus.serialize(str(file_name), corpus)
    else:
        return corpus

def save_corpus(corpus,corpus_save_name):
    """Save the corpus given copus object"""
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