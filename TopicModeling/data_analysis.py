from compute_lda import *
from manage_path import *

import pandas as pd

def merge_finra_fisd(data):
    # Get bond_issues
    print('merging data...')
    dataset_directory = get_dataset_directory()
    bond_issues_path = dataset_directory / 'Mergent_FISD_Bonds_Issues.csv'
    bond_issues_fields = ['ISSUER_ID','COMPLETE_CUSIP']
    bond_issues_dtype = {'ISSUER_ID':str , 'COMPLETE_CUSIP':str}
    bond_issues = pd.read_csv(bond_issues_path, usecols=bond_issues_fields , dtype=bond_issues_dtype)
    # Get bond_issuers
    bond_issuer_path = dataset_directory / 'Mergent_FISD_Bonds_Issuers.csv'
    bond_issuer_fields = ['ISSUER_ID', 'AGENT_ID', 'CUSIP_NAME', 'INDUSTRY_GROUP','INDUSTRY_CODE', 'PARENT_ID', 'NAICS_CODE','SIC_CODE']
    bond_issuer_dtype = {'ISSUER_ID':str, 'AGENT_ID':str, 'CUSIP_NAME':str, 'INDUSTRY_GROUP':str\
                         ,'INDUSTRY_CODE': str, 'PARENT_ID': str, 'NAICS_CODE':str, 'SIC_CODE':str}
    bond_issuer = pd.read_csv(bond_issuer_path, usecols=bond_issuer_fields, encoding='cp1252', dtype=bond_issuer_dtype)

    # Merge data with bond issues using complete cusip
    data = data.merge(bond_issues, left_on='CUSIP_ID', right_on='COMPLETE_CUSIP', how='left')
    # Then, merge data with bond issuers using ISSUER_ID
    data = data.merge(bond_issuer, left_on='ISSUER_ID', right_on='ISSUER_ID', how='left')
    print('merging data done!')
    return data

def produce_bonds_info(data):
    bonds_info = data.drop_duplicates(subset=['BOND_SYM_ID'])
    bonds_info = bonds_info[['BOND_SYM_ID', 'CUSIP_ID','ISSUER_ID', 'COMPLETE_CUSIP', 'AGENT_ID', 'CUSIP_NAME','INDUSTRY_GROUP', 'INDUSTRY_CODE', 'PARENT_ID', 'NAICS_CODE','SIC_CODE']].copy()
    bonds_info.to_csv('bonds_info.csv')

def report_basis_stat(data):
    print('shape of data = '.format(data.shape))
    print('number of unique bond = ')

def main():
    # Load data
    data = load_pickle("FINRA_TRACE_2015.pkl.zip")
    data = data.append(load_pickle("FINRA_TRACE_2014.pkl.zip"),ignore_index=True)
    produce_bonds_info(data)
    
if __name__== "__main__":
    main()