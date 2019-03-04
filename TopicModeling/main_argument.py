from compute_topic import load_corpus,load_id2word,compute_topic
import sys

def main():
    "pass command line arguments and compute_topic, used for computing LDA parallel on cluster"
    corpus = str(sys.argv[1])
    id2word = str(sys.argv[2])
    corpus_name = str(sys.argv[3])
    num_topics = int(sys.argv[4])
	
    corpus = load_corpus(corpus)
    id2word = load_id2word(id2word)
    compute_topic(corpus_name=corpus_name,corpus=corpus,num_topics=num_topics \
                  ,id2word=id2word,workers=11,chunksize=12500,passes=60,iterations=1000)

if __name__ == "__main__":
    main()