from get_data import load_data
from manage_path import *
from compute_topic import *
import sys

def main():
    data = load_data('TRACE2014_jinming.pkl')
    Tc_v1 = compute_Tc_v1(data)
    compute_corpus(Tc_v1,'Tc_v1')
    compute_id2word(Tc_v1,'Tc_v1')

if __name__ == "__main__":
    main()