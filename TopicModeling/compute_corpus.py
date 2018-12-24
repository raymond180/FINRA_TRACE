from get_data import load_data
from manage_path import *
from compute_topic import *
import sys

def main():
    data = load_data('TRACE2014_jinming.pkl')
    Dc_v1 = compute_Dc_v1(data)
    compute_corpus(Dc_v1,'Dc_v1')
    compute_id2word(Dc_v1,'Dc_v1')
    Dc_v1 = None
    Dc_v2 = compute_Dc_v2(data)
    compute_corpus(Dc_v2,'Dc_v2')
    compute_id2word(Dc_v2,'Dc_v2')
    Dc_v2 = None
    Dc_v3 = compute_Dc_v3(data)
    compute_corpus(Dc_v3,'Dc_v3')
    compute_id2word(Dc_v3,'Dc_v3')
    Dc_v3 =None
    Tc_v1 = compute_Tc_v1(data)
    compute_corpus(Tc_v1,'Tc_v1')
    compute_id2word(Tc_v1,'Tc_v1')
    Tc_v1 = None

if __name__ == "__main__":
    main()