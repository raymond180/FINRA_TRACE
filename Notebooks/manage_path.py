import os
from pathlib import Path

def get_current_directory():
    """Return path of current directory"""
    current_directory = Path.cwd()
    return current_directory

def create_directory(create_path):
    """Create directory based on create_path"""
    try:
        os.mkdir(create_path)
    except OSError:  
        print ("Creation of the directory %s failed, might already exist" % create_path)
    else:  
        print ("Successfully created the directory %s " % create_path)

def get_project_directory():
    """Return path of project root directory"""
    current_directory = Path.cwd()
    project_directory = current_directory.parents[0]
    return project_directory
    
def get_data_directory():
    """Return path of Data directory"""
    project_directory = get_project_directory()
    data_directory = project_directory / 'Data'
    return data_directory

def get_dataset_directory():
    """Return path of Data/dataset direcoty, for all FINRA_TRACE dataset"""
    data_directory = get_data_directory()
    dataset_directory = data_directory / "Dataset"
    return dataset_directory

def get_pickle_directory():
    """Return path of Data/Pickle directory"""
    data_directory = get_data_directory()
    pickle_directory = data_directory / "Pickle"
    return pickle_directory
    
def get_corpus_directory():
    """Return path of Data/Corpus directory"""
    data_directory = get_data_directory()
    corpus_directory = data_directory / "Corpus"
    return corpus_directory
    
def get_id2word_directory():
    """Return path of Data/id2word directory"""
    data_directory = get_data_directory()
    id2word_directory = data_directory / "id2word"
    return id2word_directory