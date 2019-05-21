# FINRA TRACE Data Research
This is a research project on Financial Industry Regulatory Authority (FINRA) Trade Reporting and Compliance Engine (TRACE) academic version under the supervision of Dr. Louiqa Raschid.

The purpose of the research is to study interaction and trading behavior among dealers in over-the-counter (OTC) corporate bond market. We utilize topic modeling techniques, mostly Latent Dirichlet allocation (LDA), to analyze bonds that were traded by dealer on each day. Our preliminary result shows that LDA has the flexibilty to analyze trading interaction in mutiple dimensions.

The visualization can be found here. https://raymond180.github.io/FINRA_TRACE/

## Dependency
**Required**
1. FINRA TRACE academic version
2. Python 3.7, Gensim, Pandas, Numpy, SKLearn, Matplotlib and Plotly

**Optional**
1. Cluster with SLURM, this can help you speed up computation. The shell scripts were design for that. 

## Documentation

## Resource
* [2 videos lecture on LDA by Dr. David Blei](http://videolectures.net/mlss09uk_blei_tm/)
* [The Little Book of LDA](https://ldabook.com/what-is-lda.html)
