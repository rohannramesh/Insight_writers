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
import sys
sys.path.append("/Users/rohanramesh/Documents/GitHub/Insight_writers/lib/")
from text_processing import ProcessArticle as pa
import suggestions as s
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from apiclient.discovery import build
from apiclient.errors import HttpError
import difflib


# load dataframe for writer feature space
writer_features = pd.read_pickle('/Users/rohanramesh/Documents/Insight/data_bball_writers/writer_features_USE.pickle')


# load word2vec model
kv = gensim.models.KeyedVectors.load(
    "/Users/rohanramesh/Documents/Insight/data_bball_writers/word2vec_model_kv.kv", mmap='r')
# load word2vec df for comparisons
w2v_df = pd.read_pickle('/Users/rohanramesh/Documents/Insight/data_bball_writers/word2vec_trained.pickle')

# client = MongoClient()
# mydb = client["testinsightdb"]

# for logo
image_filename = '/Users/rohanramesh/Documents/GitHub/Insight_writers/DashApp/assets/favicon.ico' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# for youtube suggestions
# team names
path_teams = '/Users/rohanramesh/Documents/GitHub/Insight_writers/DashApp/assets/nba-teams.txt'
with open(path_teams, 'r') as f:
    tmp = f.readlines()
team_names = [i.rstrip() for i in tmp]
nlp = en_core_web_sm.load()

# build api for y
# YOUTUBE_API_SERVICE_NAME = "youtube"
# YOUTUBE_API_VERSION = "v3"
# youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)


# embed_url = '//www.youtube.com/embed/o1BrK2KWifc'
# embed_url = ['<iframe width="480" height="270" src="//www.youtube.com/embed/Cj_4DupKfuk" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>']
embed_url = ['<iframe width="480" height="270" src="//www.youtube.com/embed/Cj_4DupKfuk" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>',
 '<iframe width="480" height="270" src="//www.youtube.com/embed/LgzhCXsBdpA" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>',
 '<iframe width="480" height="270" src="//www.youtube.com/embed/ui1SquTFnxo" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>']


#######################
# FUNCTIONS

def get_video_id(q, max_results,token, order="relevance", 
                 location=None, location_radius=None):

    search_response = youtube.search().list(
        q=q, type="video", pageToken=token,part="id,snippet", 
        maxResults=max_results, location=location, 
        locationRadius=location_radius, safeSearch = 'strict').execute()
    videoId = []
    title = []
    description = []
    statistics = []
    embed_url = []
    tok = search_response['nextPageToken']

    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            title.append(search_result['snippet']['title'])
            videoId.append(search_result['id']['videoId'])
            response = youtube.videos().list(part='statistics, snippet, player', 
                                 id=search_result['id']['videoId']
                                ).execute()
            description.append(response['items'][0]['snippet']['description'])
            statistics.append(response['items'][0]['statistics'])
            embed_url.append(response['items'][0]['player']['embedHtml'])
            
    ydict = {'title':title,'videoId':videoId,
              'description':description,'stats':statistics,
             'embed_url':embed_url}
    return ydict, tok

def get_Table_html(dataframe, max_rows=10):
    rows = []
    for i in range(min(len(dataframe), max_rows)):
        row = []
        for col in dataframe.columns:
            value = dataframe.iloc[i][col]
            # update this depending on which
            # columns you want to show links for
            # and what you want those links to be
            if col == 'suggested articles':
                article = Article(value)
                article.download()
                article.parse()
                article.title
                cell = html.Td(html.A(href=value, children=article.title, target='_blank'))
            else:
                cell = html.Td(children=value)
            row.append(cell)
        rows.append(html.Tr(row))
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        rows
    )

def get_video_rows(embed_url):
    rows = []
    for rr in range(0,1):
        rows.append(html.Iframe(src=embed_url[rr], width='260', height='150'))
    return html.Div([rows]) 



def generate_tables_author_content_similarity(df_authors, df_content, embed_url, max_rows=5):
	return [
	html.Div([
		html.Div([
			html.Div([
				html.Br([])]),

			html.H5(children='''Similar authors to consider:''' , 
				style={'textAlign': 'center', 'float': 'none'} ),


			get_Table_html(df_authors, max_rows=5)

			], className="four columns"),

		html.Div([
			html.Div([
				html.Br([])]),

			html.H5(children='''Similar content:''' , 
				style={'textAlign': 'left', 'float': 'none'}),

			get_Table_html(df_content, max_rows=5)

			], className="four columns"),

        # html.Div([
        #     html.Div([
        #         html.Br([])]),

        #     html.H5(children='''Video content:''' , 
        #         style={'textAlign': 'center', 'float': 'none'}),

        #     html.Iframe(src=embed_url[0], width='400', height='300'),
        #     html.Br([]),
        #     html.Iframe(src=embed_url[1], width='400', height='300'),
        #     html.Br([]),
        #     html.Iframe(src=embed_url[2], width='400', height='300'),
        #     # for iii in embed_url:
        #     # get_video_rows(embed_url)
        #     # html.Div([
        #     #     rows = []
        #     #     for rr in range(0,len(embed_url)):
        #     #         rows.append(html.Iframe(src=embed_url[rr], width='260', height='150'))
        #     #     rows])
        #     # [html.Iframe(src=iii) for iii in embed_url]

        #     ], className="four columns")

		], className="row"),

        html.Div([
            html.Div([
                html.Br([]),
                html.Br([]),
                html.Br([])]),

            html.H5(children='''Video content:''' , 
                style={'textAlign': 'left', 'float': 'none'},
                className="three columns"),

            html.Div([
                html.Div([
                    html.Br([])]),

                html.Iframe(src=embed_url[0], width='400', height='300'),

                ], className="three columns"),

            html.Div([
                html.Div([
                    html.Br([])]),

                html.Iframe(src=embed_url[1], width='400', height='300'),

                ], className="three columns"),

            html.Div([
                html.Div([
                    html.Br([])]),

                html.Iframe(src=embed_url[2], width='400', height='300'),

                ], className="three columns"),


            # html.Iframe(src=embed_url[0], width='400', height='300'),
            # html.Br([]),
            # html.Iframe(src=embed_url[1], width='400', height='300'),
            # html.Br([]),
            # html.Iframe(src=embed_url[2], width='400', height='300'),

            ], className="row")

	]


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

    html.H1(children='Full Court Presser',
	    style={'float': 'inherit'
	            # 'textAlign': 'center',
        }
        ),


	]),
    html.H2(children='''find your favorite writers.''' , 
    	style={'textAlign': 'left', 'float': 'none'}),


	html.Div([html.Br([]),
    	html.Br([])]),

    html.Div([
    	dcc.Input(id='url-input-box', type='text', size=55, 
    		placeholder='Enter url of NBA article', style= {'horizontalAlign':"left", 'float': 'none'}),
        dcc.Input(id='author-input-box', type='text', size=55, 
            placeholder='Enter favorite NBA author', style= {'horizontalAlign':"left", 'float': 'none'}),
        html.Button(id='submit-button', n_clicks=0, children='Submit'),
        html.Div(id='output-state'),
]),

        # style = dict(width = '70%', display = 'table-cell', verticalAlign = "middle",
        # 	textAlign="center"),
])

# for url
@app.callback(
    Output('output-state', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('url-input-box', 'value'),
    State('author-input-box', 'value')]
)
def update_output_div(n_clicks, input_value1, input_value2):
    # url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', input_value)
    if not input_value2:
        article, lem_txt = s.grab_article(input_value1)
        author_sugg = s.give_suggestion_featurespace_single_article(writer_features, txtstr=article.text)
        article_sugg = s.recommend_article_content(kv, w2v_df, lem_text=lem_txt)
        search_string = s.get_search_terms(article.title, team_names)
        # youtube_ouput, b = get_video_id(search_string, 1, None)
        urls_to_pass = s.parse_iframe_html(embed_url) 
        return generate_tables_author_content_similarity(author_sugg, article_sugg, urls_to_pass)
    elif not input_value1:
        author_sugg = s.give_author_suggestion_from_author(writer_features, input_value2)
        if isinstance(author_sugg, str):
            return author_sugg
        else:
            return get_Table_html(author_sugg, max_rows=5)
    else:
        article, lem_txt = s.grab_article(input_value1)
        author_sugg = s.give_author_suggestion_from_author(writer_features, input_value2)
        article_sugg = s.recommend_article_content(kv, w2v_df, lem_text=lem_txt)
        # youtube_ouput, b = get_video_id(search_string, 1, None
        return generate_tables_author_content_similarity(author_sugg, article_sugg, embed_url)





if __name__ == '__main__':
    app.run_server(debug=True)