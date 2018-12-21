from compute_topic import load_corpus,load_id2word,compute_topic

def main():
    corpus = load_corpus("matrix_1")
    id2word = load_id2word("matrix_1")
    corpus_name = "matrix_1"
    num_topics_array = [100,250,500,750]
    for num in num_topics_array:
        num_topics = num
        compute_topic(corpus_name=corpus_name,corpus=corpus,num_topics=num_topics,id2word=id2word,workers=3)

if __name__ == "__main__":
    main()