from manage_path import *
from compute_topic import *
from get_data import *

import gensim
import pyLDAvis.gensim
from gensim.corpora import Dictionary

import os
from pathlib import Path

import plotly
import plotly.graph_objs as go
import plotly.io as pio
#from itertools import repeat
from collections import deque

# The model with model_name we want to load
def load_model(model_name,num_topics):
    """Load LDAModel for analysis"""
    print('Loading LDAModel ....')
    topic_name ="_{}topics".format(num_topics)
    file_name = model_name + topic_name
    LDAModel_directory = get_LDAModel_directory()
    load_path = LDAModel_directory / "{}/{}".format(file_name,file_name)
    load_path = str(load_path)
    lda = gensim.models.ldamulticore.LdaMulticore.load(load_path)
    print('LDAModel loaded!!!')
    return lda

def document_topic_distribution(corpus,matrix_object,model,model_name,num_topics,minimum_probability=0.10):
    print('caculating document_topic_distribution ...')
    # minimum_probability is our threshold
    document_topics = model.get_document_topics(corpus,minimum_probability=minimum_probability)
    # convert document_topics, which is a gesim corpus, to numpy array
    document_topic_distribution_numpy = gensim.matutils.corpus2dense(document_topics,num_terms=int(num_topics))
    # need to transpose it because gensim represents documents on columns token on index
    document_topic_distribution_numpy = np.transpose(document_topic_distribution_numpy)
    # combine document_topic_distribution with index from matrix and columns represents gensim topics
    document_topic_distribution_pandas = pd.DataFrame(data=document_topic_distribution_numpy,index=matrix_object.index,columns=np.arange(1,int(num_topics)+1,1))
    # Only get the top three topics per document
    document_topic_distribution_pandas = document_topic_distribution_pandas[document_topic_distribution_pandas.rank(axis=1,method='max',ascending=False) <= 3]
    print('caculating document_topic_distribution done!!!')
    # Save the dataframe to csv
    print('saving document_topic_distribution...')
    result_directory = get_result_directory()
    if not result_directory.is_dir():
        create_directory(result_directory)
    file_name = result_directory / '{}_{}topics.csv'.format(model_name,num_topics)
    document_topic_distribution_pandas.to_csv(file_name)
    print('document_topic_distribution saved!!!')
    
def get_document_topic_distribution(model_name,num_topics):
    result_directory = get_result_directory()
    file_name = result_directory / '{}_{}topics.csv'.format(model_name,num_topics)
    return file_name
    
def save_pyldavis2html(model,corpus,dictionary,model_name,num_topics):
    print('preparing pyLDAvis ...')
    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary, sort_topics=False)
    print('pyLDAvis done!!!')
    print('saving pyLDAvis to html ...')
    result_directory = get_result_directory()
    if not result_directory.is_dir():
        create_directory(result_directory)
    file_name = result_directory / '{}_{}topics.html'.format(model_name,num_topics)
    # Save visualization
    pyLDAvis.save_html(vis, str(file_name))
    print('pyLDAvis to html saved!!!')
    
def get_document_item(document,position):
    return str(document).split(',')[position]

def get_dealer_by_ID(matrix,dealer_id,model_name):
    """get a subset of matrix given a dealerID"""
    result = matrix.loc[matrix['dealer'] == dealer_id].copy().drop(labels='dealer',axis=1)
    trading_days = matrix.index.unique()
    return (result,dealer_id,model_name,trading_days)

def topicXtime_plotly(topic_matrix,model_name):
    """plot topic evolution across time (topicXtime) of a dealer"""
    def topicXtime(dealer_data):
        dealer_data,dealer_id,model_name = dealer_data[0],dealer_data[1],dealer_data[2]

        heatmap_data = [
            go.Heatmap(
                z=dealer_data.values.tolist(),
                zmin=0,
                zmax=1,
                x=dealer_data.columns,
                y=dealer_data.index,
                colorscale='Jet',
            )
        ]

        layout = go.Layout(
            title='{} Dealer {}: Topic-Time'.format(model_name,dealer_id),
        )

        fig = go.Figure(data=heatmap_data, layout=layout)

        image_directory = get_image_directory() / '{}'.format(model_name)
        if not image_directory.is_dir():
            create_directory(image_directory)
        file_path = image_directory / '{}_dealer{}_topic_time.png'.format(model_name,dealer_id)
        pio.write_image(fig, str(file_path))
    
    dealer_df_list = list(map(lambda x: get_dealer_by_ID(topic_matrix,x,model_name),list(topic_matrix['dealer'].unique())))
    deque(map(topicXtime,dealer_df_list))
    
def topicXtime_plotly_parallel(dealer_data):
    """paralle version of topicXtime_plotly"""
    dealer_data,dealer_id,model_name,trading_days = dealer_data[0],dealer_data[1],dealer_data[2],dealer_data[3]
    dealer_data = dealer_data.reindex(trading_days, fill_value=np.nan).sort_index()
    
    heatmap_data = [
        go.Heatmap(
            z=dealer_data.values.tolist(),
            zmin=0,
            zmax=1,
            x=dealer_data.columns,
            y=dealer_data.index,
            colorscale='Jet',
        )
     ]

    layout = go.Layout(
        title='{} Dealer {}: Topic-Time'.format(model_name,dealer_id),
    )

    fig = go.Figure(data=heatmap_data, layout=layout)

    image_directory = get_image_directory() / '{}'.format(model_name)
    if not image_directory.is_dir():
         create_directory(image_directory)
    file_path = image_directory / '{}_dealer{}_topic_time.png'.format(model_name,dealer_id)
    pio.write_image(fig, str(file_path))
    
def topicXtime_matplotlib(dealer_data):
    """paralle plotting topicXtime using matplotlib"""
    import matplotlib.pyplot as plt
    # Get and set data
    dealer_data,dealer_id,model_name,trading_days = dealer_data[0],dealer_data[1],dealer_data[2],dealer_data[3]
    dealer_data = dealer_data.reindex(trading_days, fill_value=np.nan).sort_index()
    # Initialize figure
    fig, ax = plt.subplots()
    im = ax.imshow(dealer_data,plt.get_cmap("jet"),aspect='auto',vmin=0,vmax=1)
    # We want to show every 10 topics
    ax.set_xticks(np.arange(0,len(dealer_data.columns),10))
    # We want to show every 22 days (that's a month)
    ax.set_yticks(np.arange(0,len(dealer_data.index.date),22))
    # ... and label them with the respective list entries
    ax.set_xticklabels(dealer_data.columns[::10])
    ax.set_yticklabels(dealer_data.index.date[::22])
    # Set axis labels
    ax.set_xlabel('Topic ID', fontsize=10)
    ax.set_ylabel('Time', fontsize=10)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Set color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("probability weighting", rotation=-90, va="bottom")
    cbar.set_clim(0,1)
    # Set title
    ax.set_title('{} Dealer {}: Topic-Time'.format(model_name,dealer_id))
    fig.tight_layout()
    # Set grid
    ax.grid(which="major", color="#e5e7e9", linestyle='-', linewidth=0.5)
    # Save fig
    image_directory = get_image_directory() / '{}'.format(model_name)
    if not image_directory.is_dir():
         create_directory(image_directory)
    file_path = image_directory / '{}_dealer{}_topic_time.png'.format(model_name,dealer_id)
    fig.savefig(str(file_path),dpi=300)
    # Close fig
    plt.close(fig)

def old_topicXtime_matplotlib(df,dealer_id,matrix_name):
    """plot topicXtime of a dealer with matplotlib, used when plotly doesn't work"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates = df.index.to_pydatetime()
    dnum = mdates.date2num(dates)
    start = dnum[0] - (dnum[1]-dnum[0])/2.
    stop = dnum[-1] + (dnum[1]-dnum[0])/2.
    extent = [start, stop, -0.5, len(df.columns)-0.5]
    
    fig, ax = plt.subplots()
    im = ax.imshow(df.T.values, extent=extent, aspect="auto",vmin=0,vmax=1)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    fig.colorbar(im)
    fig.suptitle('{} Dealer {}: Topic-Time'.format(matrix_name,dealer_id), fontsize=18)
    plt.xlabel('Day', fontsize=12)
    plt.ylabel('Topic ID', fontsize=12)
    
    image_directory = get_image_directory() / '{}'.format(matrix_name)
    if not image_directory.is_dir():
        create_directory(image_directory)
    file_path = image_directory / '{}_dealer{}_topic_time.png'.format(matrix_name,dealer_id)
    plt.savefig(str(file_path), dpi=400)
    plt.close(fig)

def create_sankey_matrix(Dc_matrix,threshold,topic_selection=None):
    """Create the target to source matrix needed to plot Sankey diagram with Plotly"""
    #Get the sum of probability weighting groupby dealer divide by 250 trading days in a year
    Dc_dealerXtopic_sum = Dc_matrix.groupby(by='dealer').sum() / 250
    Dc_dealerXtopic_sum.index = Dc_dealerXtopic_sum.index.format()
    Dc_dealerXtopic_sum = pd.DataFrame(Dc_dealerXtopic_sum.stack().reset_index().rename({'level_0':'dealer','level_1':'topicID',0:'values'},axis=1))
    Dc_dealerXtopic_sum = Dc_dealerXtopic_sum[Dc_dealerXtopic_sum['values']>threshold].copy()
    Dc_dealerXtopic_sum['B/S'] = Dc_dealerXtopic_sum.apply(lambda x: str(x['dealer']).split(',')[1][1:-1] ,axis=1)
    Dc_dealerXtopic_sum['dealer'] = Dc_dealerXtopic_sum.apply(lambda x: str(x['dealer']).split(',')[0][1:] ,axis=1)
    if not topic_selection:
        pass
    else:
        Dc_dealerXtopic_sum = Dc_dealerXtopic_sum[Dc_dealerXtopic_sum['topicID'].isin([str(x) for x in topic_selection])]
    #Sort this way to allow correct order of position
    Dc_dealerXtopic_sum = Dc_dealerXtopic_sum.sort_values(by=['topicID','dealer'])
    #Encode all dealers and topics to allow correct order of position in all cases
    from sklearn import preprocessing
    dealer_le = preprocessing.LabelEncoder()
    dealer_le.fit(Dc_dealerXtopic_sum['dealer'])
    dealer_transform = dealer_le.transform(Dc_dealerXtopic_sum['dealer'])
    dealer_inverse_transform = dealer_le.inverse_transform(dealer_transform)
    topic_le = preprocessing.LabelEncoder()
    topic_le.fit(Dc_dealerXtopic_sum['topicID'])
    topic_transform = topic_le.transform(Dc_dealerXtopic_sum['topicID'])
    topic_inverse_transform = topic_le.inverse_transform(topic_transform)
    #Adjust dealer_label_position
    Dc_dealerXtopic_sum['dealer_encoding'] = dealer_transform
    topicID_size = Dc_dealerXtopic_sum['topicID'].nunique()
    Dc_dealerXtopic_sum['dealer_label_position'] = Dc_dealerXtopic_sum.apply(lambda x: x['dealer_encoding'] + topicID_size, axis=1)
    Dc_dealerXtopic_sum['topic_encoding'] = topic_transform
    Dc_dealerXtopic_sum['topic_position'] = topic_transform
    return Dc_dealerXtopic_sum

def plot_sankey(Dc_dealerXtopic_sum,title,width=1000,height=2000):
    # Create Node labels
    topic_label = list(Dc_dealerXtopic_sum['topicID'].unique())
    dealer_label = list(Dc_dealerXtopic_sum.sort_values(by=['dealer_encoding'])['dealer'].unique())
    label = []
    label.extend(topic_label)
    label.extend(dealer_label)
    # Create Node Label's Colors
    label_color = len(topic_label)*['black',]
    label_color.extend(len(dealer_label)*['black',])
    # Create Link Colors based on B/S
    link_color_dict={
        'BfD':'darkred',
        'BfC':'#0042FD', #Blue
        'StD':'#ff00ff', #Fuchsia
        'StC':'#009B00', #Green
    }
    Dc_dealerXtopic_sum['link_color'] = Dc_dealerXtopic_sum['B/S'].apply(lambda x:link_color_dict[x])

    data = dict(
        type='sankey',
        orientation = "h",
        valueformat = ".4f",
        node = dict(
          pad = 100,
          thickness = 30,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = label,
          color = label_color
        ),
        link = dict(
          source = Dc_dealerXtopic_sum['dealer_label_position'],
          target = Dc_dealerXtopic_sum['topic_position'],
          value = Dc_dealerXtopic_sum['values'],
          #label = inverse_transform
          color = Dc_dealerXtopic_sum['link_color']
      ))

    #title = str("Dc_v4_75topics_THRESHOLD={}").format(threshold)
    layout =  dict(
        title = title,
        font = dict(
          size = 20
        ),
        width=width,#750
        height=height,#1000
    )

    fig = dict(data=[data], layout=layout)
    #plotly.offline.iplot(fig, validate=False)
    pio.write_image(fig, "{}.png".format(title))
    plotly.offline.plot(fig, filename =  "{}.html".format(title), auto_open=False)
    
def main():
    model_name = str(sys.argv[1])
    num_topics = int(sys.argv[2])
    # Fro visualization
    corpus = load_corpus(model_name)
    id2word = load_id2word(model_name)
    dictionary = Dictionary.from_corpus(corpus,id2word=id2word)
    # Load LDAModel
    model = load_model(model_name,num_topics)
    save_pyldavis2html(model, corpus, dictionary,model_name,num_topics)

    # Load data to caculate matrix
    data = load_data()
    if (model_name=='Dc_v1'):
        matrix_object = compute_Dc_v1(data)
    if (model_name=='Dc_v2'):
        matrix_object = compute_Dc_v2(data)
    if (model_name=='Dc_v3'):
        matrix_object = compute_Dc_v3(data)
    if (model_name=='Dc_v4'):
        matrix_object = compute_Dc_v4(data)
    if (model_name=='Tc_v1'):
        matrix_object = compute_Tc_v1(data)
	# Save document_topic_distribution
    document_topic_distribution(corpus,matrix_object,model,model_name,num_topics)

if __name__ == "__main__":
    main()