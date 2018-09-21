import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from math import log10, floor
from dash.dependencies import Input, Output, State
# import pymongo
# from pymongo import MongoClient
from newspaper import Article
from string import digits
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from numpy import dot
from numpy.linalg import norm
from scipy.stats import skew
import math
import re
import numpy as np
import base64
import gensim
from nltk.corpus import wordnet # To get words in dictionary with their parts of speech
from nltk.stem import WordNetLemmatizer # lemmatizes word based on it's parts of speech
from collections import Counter 


# load dataframe for writer feature space
writer_features = pd.read_pickle('/Users/rohanramesh/Documents/Insight/data_bball_writers/writer_features.pickle')

# load word2vec model
kv = gensim.models.KeyedVectors.load(
    "/Users/rohanramesh/Documents/Insight/data_bball_writers/word2vec_model_kv.kv", mmap='r')
# load word2vec df for comparisons
w2v_df = pd.read_pickle('/Users/rohanramesh/Documents/Insight/data_bball_writers/word2vec_trained.pickle')

# client = MongoClient()
# mydb = client["testinsightdb"]

image_filename = '/Users/rohanramesh/Documents/GitHub/Insight_writers/DashApp/assets/bball.jpeg' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

#######################
# FUNCTIONS
def get_pos( word ):
    w_synsets = wordnet.synsets(word)

    pos_counts = Counter()
    pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )
    
    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0] # first indexer for 
            # getting the top POS from list, second indexer for getting POS from tuple( POS: count )


def lemstr(txtstr):
    lz = WordNetLemmatizer()
    newstr = [lz.lemmatize(curr_word, get_pos(curr_word)) for curr_word in txtstr]
    return newstr

def generate_table_author_similarity(dataframe, max_rows=10):
    return [
	html.Div([
    	html.Br([])]),

    html.H5(children='''Similar authors to consider:''' , 
    	style={'textAlign': 'left', 'float': 'none'}),


    html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))])]


def get_Table_html(dataframe, max_rows=10):
    rows = []
    for i in range(min(len(dataframe), max_rows)):
        row = []
        for col in dataframe.columns:
            value = dataframe.iloc[i][col]
            # update this depending on which
            # columns you want to show links for
            # and what you want those links to be
            if col == 'url':
                cell = html.Td(html.A(href=value, children=value, target='_blank'))
            else:
                cell = html.Td(children=value)
            row.append(cell)
        rows.append(html.Tr(row))
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        rows
    )



def generate_tables_author_content_similarity(df_authors, df_content, max_rows=5):
	return [
	html.Div([
		html.Div([
			html.Div([
				html.Br([])]),

			html.H5(children='''Similar authors to consider:''' , 
				style={'textAlign': 'center', 'float': 'none'}),


			get_Table_html(df_authors, max_rows=5)

			], className="six columns"),

		html.Div([
			html.Div([
				html.Br([])]),

			html.H5(children='''Similar content:''' , 
				style={'textAlign': 'center', 'float': 'none'}),

			get_Table_html(df_content, max_rows=5)

			], className="six columns")

		], className="row")

	]


# functions to clean
def clean_text(txtstr):
    txtstr = re.sub(r'\n\s*\n', '', txtstr) # extra lines
    txtstr = re.sub(r'[^\w\s]','',txtstr) # punctuation
    # for numbers
#     txtstr = re.sub(r'[^A-Za-z0-9(),!?@\'\`\"\_\n]', ' ', txtstr)
#     txtstr = re.sub(r'\w*\d\w*','',txtstr) # words with number plust letters
#     remove_digits = str.maketrans('', '', digits)
#     txtstr = txtstr.translate(remove_digits)
    txtstr = re.sub(' +',' ',txtstr) # extra white spaces
    txtstr = txtstr.lower()
    return txtstr

def grab_article(article_url):
	article = Article(article_url)
	article.download()
	article.parse()
	article.text
	lem_art = lemstr(gensim.utils.simple_preprocess(article.text))
	return article, lem_art

# get number of words
def get_nwords(txtstr):
    b = word_tokenize(txtstr)
    n_words = len(b)
    return n_words

# get number of sentences and nwords per sentences and sentence length variability
def get_nsentences(txtstr):
    b = sent_tokenize(txtstr)
    n_sentences = len(b)
    nwords = []
    for curr_sent in b:
        nwords.append(get_nwords(curr_sent))
    return n_sentences, np.mean(nwords), np.std(nwords)

# number of words and word length variability
def get_word_length(txtsrt):
    tokens = word_tokenize(txtsrt)
    # Filter out punctuation
    no_punct_tokens = ([token for token in tokens 
                                            if any(c.isalpha() for c in token)])
    # Get a distribution of token lengths
    token_lengths = [len(token) for token in no_punct_tokens]
    return np.mean(token_lengths), np.std(token_lengths), skew(token_lengths)

# cosine similarity
def cos_sim(vec1,vec2):
    a = dot(vec1, vec2)/(norm(vec1)*norm(vec2))
    return a

def get_sentiments_of_sentences(txtstr):
    analyzer = SentimentIntensityAnalyzer()
    avg_sentim = []
    sentim_var = []
    sentim_all = {}
    sentim_all['neg'] = []
    sentim_all['neu'] = []
    sentim_all['pos'] = []
    sentim_all['compound'] = []
    sentences = sent_tokenize(txtstr)
    for i in sentences:
        vs = analyzer.polarity_scores(i)
        sentim_all['neg'].append(vs['neg'])
        sentim_all['neu'].append(vs['neu'])
        sentim_all['pos'].append(vs['pos'])
        sentim_all['compound'].append(vs['compound'])
    # take avg sentim for each story by averaging sentim for each sentence
    avg_sentim.append([np.mean(sentim_all['neg']), np.mean(sentim_all['neu']), 
                     np.mean(sentim_all['pos']), np.mean(sentim_all['compound'])])
    # take std of sentim for each story to measure sentiment variability
    sentim_var.append([np.std(sentim_all['neg']), np.std(sentim_all['neu']), 
                     np.std(sentim_all['pos']), np.std(sentim_all['compound'])])
    return avg_sentim, sentim_var

def build_feature_vector_for_article(txtstr):
    cleantext = clean_text(txtstr)
    n_words = get_nwords(cleantext)
    n_sentences, n_wordspersentence, n_wordspersent_variability = get_nsentences(txtstr)
    a, b = get_sentiments_of_sentences(txtstr)
    neg_sent = a[0][0]
    neu_sent = a[0][1]
    pos_sent = a[0][2]
    neg_sent_var = b[0][0]
    neu_sent_var = b[0][1]
    pos_sent_var = b[0][2]
    wordlength, wordlength_var, wordlength_skew = get_word_length(txtstr)
    output_vec = [n_words, neg_sent, neu_sent, pos_sent, neg_sent_var, 
                 neu_sent_var, pos_sent_var, n_sentences, 
                 n_wordspersentence, n_wordspersent_variability, 
                 wordlength, wordlength_var, wordlength_skew]
    return output_vec

def give_author_suggestion_from_author(writer_feature_subsection, author):
    norm_feature_df = normalize_df(writer_feature_subsection)
    authors = norm_feature_df['author_list'].tolist()
    if author not in authors:
        return 'blah'
    else:
        idx = authors.index(author)
        norm_vec = norm_feature_df.iloc[idx][:-1]
        # do similarity test
        similarity_vec = []
        for i in range(0,len(authors)):
            vec1 = norm_feature_df.iloc[i,:-1].values
            result1 = cos_sim(vec1, norm_vec)
            similarity_vec.append(round(result1*10,2)) # multiply bu 10 to scale
        tdf = pd.DataFrame.from_dict({'similarity': similarity_vec, 'authors': authors})
        output_df = tdf.sort_values(by='similarity', ascending=False)
        return output_df.iloc[0:5]



def give_suggestion_featurespace_single_article(writer_feature_subsection, txtstr=None, url=None):
    if txtstr is not None:
        vec = build_feature_vector_for_article(txtstr)
    else:
        arti = grab_article(url)
        vec = build_feature_vector_for_article(arti.text)        
    norm_vec = normalize_vec(vec, writer_feature_subsection.mean().tolist(), 
                             writer_feature_subsection.std().tolist())
    norm_feature_df = normalize_df(writer_feature_subsection)
    authors = norm_feature_df['author_list']
    # do similarity test
    similarity_vec = []
    for i in range(0,len(authors)):
        vec1 = norm_feature_df.iloc[i,:-1].values
        result1 = cos_sim(vec1, norm_vec)
        similarity_vec.append(round(result1*10,2)) # multiply bu 10 to scale
    tdf = pd.DataFrame.from_dict({'similarity': similarity_vec, 'authors': authors})
    output_df = tdf.sort_values(by='similarity', ascending=False)
    return output_df.iloc[0:5]

def normalize_df(df):
    result = df.copy()
    for feature_name in df.columns:
        if isinstance(df[feature_name][0], str):
            continue
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
#         result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        result[feature_name] = (df[feature_name] - df[feature_name].mean()) / (df[feature_name].std())
#         result[feature_name] = (df[feature_name]) / norm(df[feature_name])
    return result

# same normalization but on vector
def normalize_vec(list1, means, var):
    output_vec = [(list1[i]-means[i])/var[i] for i in range(0,len(list1))]
    return output_vec

def grab_article_for_w2v(article_url):
	article = Article(article_url)
	article.download()
	article.parse()
	txtstr = article.text
	lem_art = lemstr(gensim.utils.simple_preprocess(txtstr))
	return lem_art

# txtstr must be tokenized and lemmatized
def get_vector_from_w2v_model(keyedvectors, txtstr):
    a = [r for r in txtstr if r in keyedvectors.vocab]
    array_for_art = np.ndarray(shape=[len(a),keyedvectors.vector_size])
    for i in range(0,len(a)):
        array_for_art[i,:] = keyedvectors[a[i]]
    article_vector = np.mean(array_for_art, axis=0)
    return article_vector
    
def recommend_article_content(keyedvectors, w2v_df, lem_text=None, article_url=None):
    if article_url is not None:
        arti, a = grab_article(url)
    # preprocess data
    # a = grab_article_for_w2v(article_url)
    # get vector
    article_vector = get_vector_from_w2v_model(keyedvectors, lem_text)
    # check against current df of articles using cos_sim
    cos_sim_output = []
    for i in w2v_df:
        cos_sim_output.append(round(cos_sim(w2v_df[i],article_vector)*10,2)) # multiply by 10 for scale
    r = pd.DataFrame.from_dict({
        'similarity': cos_sim_output, "url": w2v_df.keys()
    })
    df_content_sim = r.sort_values(by='similarity', ascending=False)
    return df_content_sim

####################
app = dash.Dash(__name__)
# app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})


colors = {
    'background': '#111111',
    'text': '#111111'
}

app.layout = html.Div(children=[
	html.Div([
	html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width= 200, height = 200,
			style={'float': 'left'}),

		# style={'verticalAlign': 'middle'}),

	html.Div([
    	html.Br([])]),

    html.H1(children='Columns of the NBA',
	    style={'float': 'inherit'
	            # 'textAlign': 'center',
        }
        ),
	]),
    html.H2(children='''find your favorite writers.''' , 
    	style={'textAlign': 'left', 'float': 'none'}),

 #    dcc.Input(
	#     placeholder='Enter a url',
	#     type='text',
	#     value='',
	#     size=50,
	# 	style={'textAlign': 'center'},
	# ),
	html.Div([html.Br([]),
    	html.Br([])]),

    html.Div([
    	dcc.Input(id='url-input-box', type='text', size=130, 
    		placeholder='Enter url of NBA article', style= {'horizontalAlign':"left", 'float': 'none'}),
    	html.Button(id='submit-button', n_clicks=0, children='Submit'),
        html.Div(id='output-state'),

        dcc.Input(id='author-input-box', type='text', size=45, 
            placeholder='Enter favorite NBA author', style= {'horizontalAlign':"left", 'float': 'none'}),
        html.Button(id='submit-button2', n_clicks=0, children='Submit'),
        html.Div(id='output-state-author'),
]),

        # style = dict(width = '70%', display = 'table-cell', verticalAlign = "middle",
        # 	textAlign="center"),
])

# for url
@app.callback(
    Output('output-state', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('url-input-box', 'value')]
)
def update_output_div(n_clicks, input_value):
	article, lem_txt = grab_article(input_value)
	author_sugg = give_suggestion_featurespace_single_article(writer_features, txtstr=article.text)
	article_sugg = recommend_article_content(kv, w2v_df, lem_text=lem_txt)
	return generate_tables_author_content_similarity(author_sugg, article_sugg)

# for author
@app.callback(
    Output('output-state-author', 'children'),
    [Input('submit-button2', 'n_clicks')],
    [State('author-input-box', 'value')]
)
def update_output_div_author(n_clicks, input_value):
    author_sugg = give_author_suggestion_from_author(writer_features, input_value)
    # article, lem_txt = grab_article(input_value)
    # author_sugg = give_suggestion_featurespace_single_article(writer_features, txtstr=article.text)
    # article_sugg = recommend_article_content(kv, w2v_df, lem_text=lem_txt)
    return get_Table_html(author_sugg)


if __name__ == '__main__':
    app.run_server(debug=True)