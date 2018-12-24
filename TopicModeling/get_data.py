# Import Library
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

from manage_path import *

def read_data(file_name,low_memory=False,memory_map=True,engine='c'):
    """Read FINRA TRACE data and perform date conversion and data merging with Mergent FISD"""
    # Prepare data file path
    dataset_directory = get_dataset_directory()
    file_path = dataset_directory / file_name
    
    # Only get the field we want
    field_of_interest_datetime = ['BOND_SYM_ID','CUSIP_ID','SCRTY_TYPE_CD','ENTRD_VOL_QT','RPTD_PR','RPT_SIDE_CD' \
                                  ,'TRD_EXCTN_DT','EXCTN_TM','TRD_RPT_DT','TRD_RPT_TM', 'Report_Dealer_Index'\
                                  ,'Contra_Party_Index','TRC_ST']

    data_dtype={'BOND_SYM_ID': str, 'CUSIP_ID': str,'SCRTY_TYPE_CD':str, 'ENTRD_VOL_QT': np.float64, 'RPTD_PR': np.float32 \
           ,'RPT_SIDE_CD':str, 'Report_Dealer_Index': str,'Contra_Party_Index': str, 'TRC_ST':str}

    parse_dates = {'TRD_RPT_DTTM':['TRD_RPT_DT','TRD_RPT_TM'],'TRD_EXCTN_DTTM':['TRD_EXCTN_DT','EXCTN_TM']}
    date_parser = lambda x: pd.to_datetime(x, format='%Y%m%d %H%M%S', errors='coerce')
    
    print('reading data...')
    data = pd.read_csv(file_path, usecols=field_of_interest_datetime, dtype=data_dtype, parse_dates=parse_dates\
                       , infer_datetime_format=True, low_memory=low_memory, memory_map=memory_map, engine=engine \
                       , date_parser=date_parser)
    print('data reading done!')
    #converters={'TRD_RPT_TM':lambda x : pd.to_datetime(x,format='%H%M%S')}
    
    print('transforming data...')
    # Drop TRD_EXCTN_DTTM that is NaT
    n_drop_rows = data['TRD_EXCTN_DTTM'].size - data['TRD_EXCTN_DTTM'].count()
    data.dropna(subset=['TRD_EXCTN_DTTM'],inplace=True)
    print('{} of rows are dropped'.format(n_drop_rows))
    # Add new column document_date which is the date of TRD_EXCTN_DTTM
    data['document_date'] = data['TRD_EXCTN_DTTM'].dt.date.apply(str)
    print('transforming data done!!')
    
    # Get bond_issues
    print('merging data...')
    bond_issues_path = dataset_directory / 'Mergent_FISD_Bonds_Issues.csv'
    bond_issues_fields = ['ISSUER_ID','COMPLETE_CUSIP']
    bond_issues_dtype = {'ISSUER_ID':str , 'COMPLETE_CUSIP':str}
    bond_issues = pd.read_csv(bond_issues_path, usecols=bond_issues_fields , dtype=bond_issues_dtype \
                             , low_memory=low_memory, memory_map=memory_map)
    # Get bond_issuers
    bond_issuer_path = dataset_directory / 'Mergent_FISD_Bonds_Issuers.csv'
    bond_issuer_fields = ['ISSUER_ID', 'AGENT_ID', 'CUSIP_NAME', 'INDUSTRY_GROUP','INDUSTRY_CODE', 'PARENT_ID', 'NAICS_CODE','SIC_CODE']
    bond_issuer_dtype = {'ISSUER_ID':str, 'AGENT_ID':str, 'CUSIP_NAME':str, 'INDUSTRY_GROUP':str \
                         ,'INDUSTRY_CODE': str, 'PARENT_ID': str, 'NAICS_CODE':str, 'SIC_CODE':str}
    bond_issuer = pd.read_csv(bond_issuer_path, usecols=bond_issuer_fields, encoding='cp1252', dtype=bond_issuer_dtype \
                             , low_memory=low_memory, memory_map=memory_map)
    
    #bond_ratings_path = dataset_directory / 'Mergent_FISD_Bonds_Ratings.csv'
    #bond_ratings = pd.read_csv(bond_ratings_path)

    # Merge data with bond issues using complete cusip
    data = data.merge(bond_issues, left_on='CUSIP_ID', right_on='COMPLETE_CUSIP', how='left')
    # Then, merge data with bond issuers using ISSUER_ID
    data = data.merge(bond_issuer, left_on='ISSUER_ID', right_on='ISSUER_ID', how='left')
    print('merging data done!')
    return data

def save_to_pickle(data,pickle_name):
    """save data to pickle directory"""
    print('saving data to pickle ...')
    # get pickle directory
    pickle_directory = get_pickle_directory()
    # create directory if not exist
    if not pickle_directory.is_dir():
        create_directory(pickle_directory)
    # concatonate pickle directory and pickle name
    file_path = pickle_directory / pickle_name
    # data to pickle
    data.to_pickle(file_path)
    print('saving data to pickle done!!!')
    
def load_data(file_name="TRACE2014_jinming.pkl"):
    print("loading data {}...".format(file_name))
    pickle_directory = get_pickle_directory()
    
    pickle_file_path = pickle_directory / file_name
    
    print("Getting data from{}...".format(pickle_file_path))
    data = pd.read_pickle(pickle_file_path)
    print("Data getting success!")
    return data

def main():
    # Get file_name and pickle_name to be saved as
    file_name = str(sys.argv[1])
    pickle_name = str(sys.argv[2])
    
    print('Preparing data ...')
    data = read_data(file_name)
    save_to_pickle(data,pickle_name)
    
if __name__== "__main__":
    main()