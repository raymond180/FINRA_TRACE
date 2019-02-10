import pandas as pd

from manage_path import *
from topic_model_analysis import *

import sys

def main():
    model_name = str(sys.argv[1])
    num_topics = str(sys.argv[2])
    file_name = get_document_topic_distribution(model_name,num_topics)
    topic_matrix = pd.read_csv(file_name,index_col=0)

    get_document_item_vectorize = np.vectorize(get_document_item)
    if (model_name=='Dc_v1'):
        topic_matrix['dealer'] = get_document_item_vectorize(topic_matrix.index,0)
        topic_matrix.index = pd.to_datetime(get_document_item_vectorize(topic_matrix.index,1))
    if (model_name=='Dc_v2'):
        topic_matrix['dealer'] = pd.Series(list(zip(get_document_item_vectorize(topic_matrix.index,0),get_document_item_vectorize(topic_matrix.index,2)))).values
        topic_matrix.index = pd.to_datetime(get_document_item_vectorize(topic_matrix.index,1))
    if (model_name=='Dc_v3'):
        topic_matrix['dealer'] = pd.Series(list(zip(get_document_item_vectorize(topic_matrix.index,0),get_document_item_vectorize(topic_matrix.index,2)))).values
        topic_matrix.index = pd.to_datetime(get_document_item_vectorize(topic_matrix.index,1))
    if (model_name=='Tc_v1'):
        topic_matrix['dealer'] = pd.Series(list(zip(get_document_item_vectorize(topic_matrix.index,0),get_document_item_vectorize(topic_matrix.index,1)))).values
        topic_matrix.index = pd.to_datetime(get_document_item_vectorize(topic_matrix.index,2))

    topicXtime_vusualize(topic_matrix,model_name)

if __name__ == "__main__":
    main()