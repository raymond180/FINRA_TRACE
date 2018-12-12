# Import Library
import pandas as pd
import numpy as np
import os
from pathlib import Path

from manage_path import *

def read_data(file_name):
    dataset_directory = get_dataset_directory()
    file_path = dataset_directory / file_name
    # Only get the field we want
    field_of_interest_datetime = ['BOND_SYM_ID','CUSIP_ID','SCRTY_TYPE_CD','ENTRD_VOL_QT','RPTD_PR','RPT_SIDE_CD' \
                                  ,'TRD_EXCTN_DT_D','EXCTN_TM_D','TRD_RPT_DT','TRD_RPT_TM', 'Report_Dealer_Index'\
                                  ,'Contra_Party_Index','TRC_ST']

    dtype={'BOND_SYM_ID': str, 'CUSIP_ID': str,'SCRTY_TYPE_CD':str, 'ENTRD_VOL_QT': np.float64, 'RPTD_PR': np.float64 \
           ,'RPT_SIDE_CD':str, 'Report_Dealer_Index': str,'Contra_Party_Index': str, 'TRC_ST':str}

    parse_dates = {'TRD_RPT_DTTM':['TRD_RPT_DT','TRD_RPT_TM'],'TRD_EXCTN_DTTM':['TRD_EXCTN_DT_D','EXCTN_TM_D']}

    data = pd.read_csv(file_path,usecols=field_of_interest_datetime,parse_dates=parse_dates\
                       ,infer_datetime_format=True,converters={'TRD_RPT_TM':lambda x : pd.to_datetime(x,format='%H%M%S')})
    
    # Add new column document to concatenate Report_Dealer_Index and TRD_RPT_DTTM
    data['document_date'] = data['TRD_RPT_DTTM'].dt.date.apply(str)
    data['document'] = data.apply(lambda x: str(x['Report_Dealer_Index'])+ ',' +str(x['document_date']) ,axis=1)
    return data

def save_to_pickle(data,pickle_name):
    """save data to pickle directory"""
    # get pickle directory
    pickle_directory = get_Pickle_directory()
    # create directory if not exist
    if not pickle_directory.is_dir():
        create_directory(pickle_directory)
    # concatonate pickle directory and pickle name
    file_path = pickle_directory / pickle_name
    # data to pickle
    data.to_pickle(file_path)
    
def load_data(file_name="TRACE2014_jinming"):
    print("loading data {}...".format(file_name))
    pickle_directory = get_Pickle_directory()
    
    pickle_file_path = pickle_directory / file_name
    
    print("Getting data from{}...".format(pickle_file_path))
    data = pd.read_pickle(pickle_file_path)
    print("Data getting success!")
    return data