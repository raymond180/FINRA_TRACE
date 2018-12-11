# Import Library
import pandas as pd
import numpy as np
import os
from pathlib import Path

from manage_path import get_current_directory,create_directory

def read_data():
    root_folder = Path('../Data/')
    file_name = 'TRACE2014_jinming.csv'

    file_path = root_folder / file_name
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

def to_pickle():
    root_folder = Path('../Data/Pickle/')
    file_name = 'TRACE2014_jinming'
    file_path = root_folder / file_name
    data.to_pickle(file_path)
    
def load_data(file_name="TRACE2014_jinming"):
    print("loading data {}...".format(file_name))
    current_path = get_current_directory()
    data_save_path = current_path.parent / "./Data/Pickle/"
    data_save_path = data_save_path / file_name
    
    print("Getting data from{}...".format(data_save_path))
    data = pd.read_pickle(data_save_path)
    print("Data getting success!")
    return data