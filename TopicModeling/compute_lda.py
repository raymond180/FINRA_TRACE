# Import ML Library
import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.test.utils import datapath
import pyLDAvis.gensim
from gensim.corpora import Dictionary
# Import helper Library
import logging
from pathlib import Path
import os
import sys
# Import local files
from get_data import load_data
from manage_path import *

def load_pickle(file_name="FINRA_TRACE_2014.pkl.zip",is_zip=True):
    pickle_file_path = get_pickle_directory() / file_name
    print("Getting data from{}...".format(pickle_file_path))
    if is_zip:
        data = pd.read_pickle(pickle_file_path,compression='zip')
    else:
        data = pd.read_pickle(pickle_file_path)
    print("Data getting success from {}!".format(pickle_file_path))
    return data

def create_document_2(first,second):
    return str(first) + ',' + str(second)

def create_document_3(first,second,third):
    return str(first) + ',' + str(second) + ',' + str(third)

def document_date2year(date):
    return str(date[0:4])

def create_dummy_sink(Report_Dealer_Index,Contra_Party_Index):
    """transform Report_Dealer_Index that are 0 for OLD_Dc_v4"""
    if str(Report_Dealer_Index) == '0':
        return 'D' + str(Contra_Party_Index)
    else:
        return str(Report_Dealer_Index)
    
def create_dummy_source(Report_Dealer_Index,Contra_Party_Index):
    """transform Contra_Party_Index that are 99999 for OLD_Dc_v4"""
    if str(Contra_Party_Index) == '99999':
        return 'D' + str(Report_Dealer_Index)
    else:
        return str(Contra_Party_Index)
    
def client_to_delete(document):
    if str(document.split(',')[0][0:]) == '0' or str(document.split(',')[0][0:]) == '99999':
        return 'delete'
    else:
        return 'keep'
    
def create_buy_document(Report_Dealer_Index,Contra_Party_Index,document_date):
    if str(Report_Dealer_Index) == '0':
        return str(Contra_Party_Index) + ',' + str(document_date) + ',' + 'BfC'
    else:
        return str(Contra_Party_Index) + ',' + str(document_date) + ',' + 'BfD'

def create_sell_document(Report_Dealer_Index,Contra_Party_Index,document_date):
    if str(Contra_Party_Index) == '99999':
        return str(Report_Dealer_Index) + ',' + str(document_date) + ',' + 'StC'
    else:
        return str(Report_Dealer_Index) + ',' + str(document_date) + ',' + 'StD'
    
def create_buy_document_no_source(Report_Dealer_Index,Contra_Party_Index,document_date):
    if str(Report_Dealer_Index) == '0':
        return str(Contra_Party_Index) + ',' + str(document_date) + ',' + 'BfC'
    elif str(Contra_Party_Index) == '99999':
        return np.nan
    else:
        return str(Contra_Party_Index) + ',' + str(document_date) + ',' + 'BfD'

def create_sell_document_no_source(Report_Dealer_Index,Contra_Party_Index,document_date):
    if str(Contra_Party_Index) == '99999':
        return str(Report_Dealer_Index) + ',' + str(document_date) + ',' + 'StC'
    elif str(Report_Dealer_Index) == '0':
        return np.nan
    else:
        return str(Report_Dealer_Index) + ',' + str(document_date) + ',' + 'StD'

def compute_Dc_v1(data):
    """Compute Dc_v1 which is count of bonds on given dealer and day"""
    create_document_vectorize = np.vectorize(create_document_2)
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
    create_document_vectorize = np.vectorize(create_document_3)
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
    """Compute Dc_v3 which is count of bonds on given dealer and day seperated buy and sell"""
    create_document_vectorize = np.vectorize(create_document_3)
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

def trade_vol_BoW(data,cap="large"):
    """Compute Dc_v4 which is count of bonds on given dealer and day seperated buy and sell"""
    data['price'] = (data['ENTRD_VOL_QT'] * data['RPTD_PR'])/100
    cap_threshold = 10000
    if cap=="large":
        data = data[data['price'] >= cap_threshold]
    else:
        data = data[data['price'] < cap_threshold]
    data['document_date'] = data['TRD_EXCTN_DTTM'].dt.date.apply(lambda x: str(x))
    create_buy_document_no_source_vectorize = np.vectorize(create_buy_document_no_source)
    create_sell_document_no_source_vectorize = np.vectorize(create_sell_document_no_source)
    client_to_delete_vectorize = np.vectorize(client_to_delete)
    print("creating documents ......")
    # Add new column Dc_v4_S which is the string representation of report dealer buy on the specific day
    data['trade_vol_BoW_S'] = create_sell_document_no_source_vectorize(data['Report_Dealer_Index'].values,data['Contra_Party_Index'].values,data['document_date'].values)
    # Add new column Dc_v4_B which is the string representation of report dealer sell on the specific day
    data['trade_vol_BoW_B'] = create_buy_document_no_source_vectorize(data['Report_Dealer_Index'].values,data['Contra_Party_Index'].values,data['document_date'].values)
    print("documents created!!")
    
    data = data[['trade_vol_BoW_S','trade_vol_BoW_B','BOND_SYM_ID','price']].copy()
    data_gb_sell = data[data['trade_vol_BoW_S']!='nan'].groupby(by=['trade_vol_BoW_S','BOND_SYM_ID'])
    data_gb_buy = data[data['trade_vol_BoW_B']!='nan'].groupby(by=['trade_vol_BoW_B','BOND_SYM_ID'])
    
    print("computing bag_of_words ......")
    bag_of_words = data_gb_sell['price'].sum().unstack(fill_value=0)
    bag_of_words = bag_of_words.append(data_gb_buy['price'].sum().unstack(fill_value=0))
    bag_of_words = bag_of_words.sort_index(axis=1)
    print("computing bag_of_words done!")
    return bag_of_words

def OLD_compute_Dc_v4(data):
    """Compute Dc_v4 which is count of bonds on given dealer and day seperated buy and sell"""
    create_buy_document_vectorize = np.vectorize(create_buy_document)
    create_sell_document_vectorize = np.vectorize(create_sell_document)
    client_to_delete_vectorize = np.vectorize(client_to_delete)
    print("creating documents ......")
    # Add new column Dc_v4_S which is the string representation of report dealer buy on the specific day
    data['Dc_v4_S'] = create_sell_document_vectorize(data['Report_Dealer_Index'].values,data['Contra_Party_Index'].values,data['document_date'].values)
    # Add new column Dc_v4_B which is the string representation of report dealer sell on the specific day
    data['Dc_v4_B'] = create_buy_document_vectorize(data['Report_Dealer_Index'].values,data['Contra_Party_Index'].values,data['document_date'].values)
    print("documents created!!")
    
    data_gb_sell = data.groupby(by=['Dc_v4_S','BOND_SYM_ID'])
    data_gb_buy = data.groupby(by=['Dc_v4_B','BOND_SYM_ID'])
    
    print("computing Dc_v4 ......")
    Dc_v4 = data_gb_sell['BOND_SYM_ID'].size().astype(np.int16).unstack(fill_value=0)
    Dc_v4 = Dc_v4.append(data_gb_buy['BOND_SYM_ID'].size().astype(np.int16).unstack(fill_value=0))
    Dc_v4 = Dc_v4.sort_index(axis=1)
    print("computing Dc_v4 done!")
    print("flitering out general client in Dc_v4")
    Dc_v4['to_delete'] = client_to_delete_vectorize(Dc_v4.index)
    Dc_v4 = Dc_v4.loc[Dc_v4['to_delete']!='delete'].drop(['to_delete'],axis=1).copy()
    Dc_v4 = Dc_v4[Dc_v4.sum(axis=1) > 3].copy()
    Dc_v4.dropna(axis=1,how='all',inplace=True)
    print("all done!")
    return Dc_v4

def compute_Tc_v1(data):
    """Compute Tc_v1 which is a document will represent the triple (seller, bond, buyer, date) directly"""
    create_document_vectorize = np.vectorize(create_document_3)
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

def compute_corpus(bag_of_words,corpus_save_name,save=True):
    """Compute corpus given a bag_of_words and save it"""
    print("computing corpus...")
    corpus = gensim.matutils.Dense2Corpus(bag_of_words.values,documents_columns=False)
    print("corpus computed!!")
    if save:
        print("saving corpus...")
        corpus_directory = get_corpus_directory()
        if not corpus_directory.is_dir():
            create_directory(corpus_directory)
        file_name = corpus_directory / "{}.mm".format(corpus_save_name)
        gensim.corpora.MmCorpus.serialize(str(file_name), corpus)
        print("corpus saved!!")
        return corpus
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

def compute_id2word(bag_of_words,bag_of_words_name,save=True):
    """compute id2word of a bag_of_words and save(return) the id2word as dictionary using numpy"""
    print("computing id2word...")
    le = preprocessing.LabelEncoder()
    le.fit(bag_of_words.columns)
    transform = le.transform(bag_of_words.columns)
    inverse_transform = le.inverse_transform(transform)
    id2word = dict(zip(transform, inverse_transform))
    print("id2word computed!!")
    if(save):
        print("saving id2word ...")
        id2word_directory = get_id2word_directory()
        if not id2word_directory.is_dir():
            create_directory(id2word_directory)
        file_name = id2word_directory / "{}.npy".format(bag_of_words_name)
        # save the id2word using numpy
        np.save(file_name, id2word)
        print("id2word saved!!")
        return id2word
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
    return lda

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
    
# ------------------------ LDA Analysis -------------------------
def document_topic_distribution(corpus,matrix_object,model,model_name,num_topics,minimum_probability=0.10):
    print('caculating document_topic_distribution ...')
    # minimum_probability is our threshold
    document_topics = model.get_document_topics(corpus,minimum_probability=minimum_probability)
    # convert document_topics, which is a gesim corpus, to numpy array
    document_topic_distribution_numpy = gensim.matutils.corpus2dense(document_topics,num_terms=int(num_topics))
    # need to transpose it because gensim represents documents on columns token on index
    document_topic_distribution_numpy = np.transpose(document_topic_distribution_numpy)
    # combine document_topic_distribution with index from matrix and columns represents gensim topics
    document_topic_distribution_pandas = pd.DataFrame(data=document_topic_distribution_numpy,index=matrix_object.index,columns=np.arange(1,int(num_topics)+1,1))
    # Only get the top three topics per document
    document_topic_distribution_pandas = document_topic_distribution_pandas[document_topic_distribution_pandas.rank(axis=1,method='max',ascending=False) <= 3]
    print('caculating document_topic_distribution done!!!')
    # Save the dataframe to csv
    print('saving document_topic_distribution...')
    result_directory = get_result_directory()
    if not result_directory.is_dir():
        create_directory(result_directory)
    file_name = result_directory / '{}_{}topics.csv'.format(model_name,num_topics)
    document_topic_distribution_pandas.to_csv(file_name)
    print('document_topic_distribution saved!!!')
    
def save_pyldavis2html(model,corpus,dictionary,model_name,num_topics):
    print('preparing pyLDAvis ...')
    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary, sort_topics=False)
    print('pyLDAvis done!!!')
    print('saving pyLDAvis to html ...')
    result_directory = get_result_directory()
    if not result_directory.is_dir():
        create_directory(result_directory)
    file_name = result_directory / '{}_{}topics.html'.format(model_name,num_topics)
    # Save visualization
    pyLDAvis.save_html(vis, str(file_name))
    print('pyLDAvis to html saved!!!')

def get_document_topic_distribution(model_name,num_topics):
    result_directory = get_result_directory()
    file_name = result_directory / '{}_{}topics.csv'.format(model_name,num_topics)
    return file_name

def get_document_item(document,position):
    return str(document).split(',')[position]

def get_dealer_by_ID(matrix,dealer_id,model_name):
    """get a subset of matrix given a dealerID"""
    result = matrix.loc[matrix['dealer'] == dealer_id].copy().drop(labels='dealer',axis=1)
    trading_days = matrix.index.unique()
    return (result,dealer_id,model_name,trading_days)
def topicXtime_matplotlib(dealer_data):
    """paralle plotting topicXtime using matplotlib"""
    import matplotlib.pyplot as plt
    # Get and set data
    dealer_data,dealer_id,model_name,trading_days = dealer_data[0],dealer_data[1],dealer_data[2],dealer_data[3]
    dealer_data = dealer_data.reindex(trading_days, fill_value=np.nan).sort_index()
    # Initialize figure
    fig, ax = plt.subplots()
    im = ax.imshow(dealer_data,plt.get_cmap("jet"),aspect='auto',vmin=0,vmax=1)
    # We want to show every 10 topics
    ax.set_xticks(np.arange(0,len(dealer_data.columns),10))
    # We want to show every 22 days (that's a month)
    ax.set_yticks(np.arange(0,len(dealer_data.index.date),22))
    # ... and label them with the respective list entries
    ax.set_xticklabels(dealer_data.columns[::10])
    ax.set_yticklabels(dealer_data.index.date[::22])
    # Set axis labels
    ax.set_xlabel('Topic ID', fontsize=10)
    ax.set_ylabel('Time', fontsize=10)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Set color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("probability weighting", rotation=-90, va="bottom")
    cbar.set_clim(0,1)
    # Set title
    ax.set_title('{} Dealer {}: Topic-Time'.format(model_name,dealer_id))
    fig.tight_layout()
    # Set grid
    ax.grid(which="major", color="#e5e7e9", linestyle='-', linewidth=0.5)
    # Save fig
    image_directory = get_image_directory() / '{}'.format(model_name)
    if not image_directory.is_dir():
         create_directory(image_directory)
    file_path = image_directory / '{}_dealer{}_topic_time.png'.format(model_name,dealer_id)
    fig.savefig(str(file_path),dpi=300)
    # Close fig
    plt.close(fig)
    
def main():
    # ---------------- Set MLK Enviroment Variables for better Gensim LDA performance  ----------------
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    # ---------------- Prepare LDA Inputs & Run LDA ----------------
    # Parse command line args
    save_name = str(sys.argv[1])
    cap = str(sys.argv[2])
    num_topics = int(sys.argv[3])
    # Load data
    data = load_pickle("FINRA_TRACE_2015.pkl.zip")
    data = data.append(load_pickle("FINRA_TRACE_2014.pkl.zip"),ignore_index=True)
    data = data.append(load_pickle("FINRA_TRACE_2013.pkl.zip"),ignore_index=True)
    data = data.append(load_pickle("FINRA_TRACE_2012.pkl.zip"),ignore_index=True)
    # Compute a version of bag_of_words given the save_name
    if save_name=="trade_vol_BoW":
        bag_of_words = trade_vol_BoW(data,cap)
        save_name = save_name+cap
    else:
        print("the save_name does not have a corresponding bag_of_words")
    # Compute input for gensim LDA
    corpus = compute_corpus(bag_of_words,save_name)
    id2word = compute_id2word(bag_of_words,save_name)
    # Run Gensim LDA
    lda = compute_topic(save_name,corpus,num_topics,id2word)
    # ---------------- LDA Analysis  ----------------
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"
    # Run PyLDAvis
    dictionary = Dictionary.from_corpus(corpus,id2word=id2word)
    save_pyldavis2html(lda, corpus, dictionary,save_name,num_topics)
    # Save document X topic matrix to csv
    document_topic_distribution(corpus,bag_of_words,lda,save_name,num_topics)
    
if __name__== "__main__":
    main()