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
The recommended enviroment to run all the bash script without modification needs to follow these requirements:
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and install it at the default path ~/miniconda3/ so that you can activate the base conda enviroment using the below script, and that is what all the bash script contains
```
source ~/miniconda3/bin/activate
```
2. Install all the python dependences mentions above in the conda enviroment you install from the previous step. That is
```
conda install ......
``` 
3. Finally, Submit SLURM jobs using
```
sbatch any_sbatch_script.sh
``` 

### Directories:
FINRA_TRACE/


| Directories | Explaination | 
| -------- | -------- | 
| FINRA_TRACE/     | base directory     |
| FINRA_TRACE/Notebook/     | folder to place experiment notebooks (not used anymore)     |
| FINRA_TRACE/Data/     | data directory contains the following directory   |
| FINRA_TRACE/Data/Pickle/      | folder to place data in pandas Pickle format you get from the sas2csv.py (I know the naming is confusing)    |
| FINRA_TRACE/Data/Dataset/     | folder to place Mergent FISD .csv files you download from WRDS  |
| FINRA_TRACE/Data/id2word/     | folder to save Gensim id2word output from compute_lda.py     |
| FINRA_TRACE/Data/Corpus/     | folder to save Gensim corpus output from compute_lda.py          |
| FINRA_TRACE/Result/     | auto-generated folder to save the pyldaviz .html output and document-topic probability weighting matrix .csv from lda_analysis.py   |
| FINRA_TRACE/LDAModel/     | auto-generated folder to save the Gensim lda model file outputed by compute_lda.py   |
| FINRA_TRACE/LDAModel/logs/     | auto-generated folder to save the Gensim lda log files outputed by compute_lda.py (this is where you look for perplexity)  |


### Files:
| Files | Explaination | 
| -------- | -------- | 
| ./TopicModeling/manage_path.py     | just to nicely create/call the path of each directory in relative links     |
| ./TopicModeling/compute_lda.py     |  load python pandas pickle files and compute Gensim lda    |
| ./TopicModeling/compute_lda.sh     |  run compute_lda.py in cluster with each node a specified command line arguments (what kind of transformations, small/large caps ,number of topics ...)    |
| ./TopicModeling/lda_analysis.py    |  analysis on lda result inclusing document_topic_distribution, save_pyldavis2html, topicXtime_matplotlib, plot_sankey   |
| ./TopicModeling/lda_analysis.sh    |  sbatch script to run lda analysis on cluster |
| ./TopicModeling/topicXtime.py    |  plot topicXtime visualization |
| ./TopicModeling/topicXtime.sh    |  plot topicXtime visualization on cluster as sbatch job |



## Resource
* [2 videos lecture on LDA by Dr. David Blei](http://videolectures.net/mlss09uk_blei_tm/)
* [The Little Book of LDA](https://ldabook.com/what-is-lda.html)
