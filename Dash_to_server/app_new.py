import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from math import log10, floor
from dash.dependencies import Input, Output, State
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
import pickle
from nltk.corpus import wordnet # To get words in dictionary with their parts of speech
from nltk.stem import WordNetLemmatizer # lemmatizes word based on it's parts of speech
from collections import Counter 
import sys
from text_processing import ProcessArticle as pa
import suggestions as s
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from apiclient.discovery import build
from apiclient.errors import HttpError
import difflib
import time


# load dataframe for writer feature space
writer_features = pd.read_pickle('writer_features_USE.pickle')


# load word2vec model
kv = gensim.models.KeyedVectors.load(
    "word2vec_model_kv.kv", mmap='r')
# load word2vec df for comparisons
w2v_df = pd.read_pickle('word2vec_trained.pickle')

# client = MongoClient()
# mydb = client["testinsightdb"]

# for logo
image_filename = 'BlackWhiteLogo.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# for background image
# bg_image_filename = '/Users/rohanramesh/Documents/Insight/data_bball_writers/NBA_court.jpg' # replace with your own image
# encoded_image_bg = base64.b64encode(open(bg_image_filename, 'rb').read())

# for youtube suggestions
# team names
path_teams = 'nba-teams.txt'
with open(path_teams, 'r') as f:
    tmp = f.readlines()
team_names = [i.rstrip() for i in tmp]
nlp = en_core_web_sm.load()

# config file
path_teams = 'config.txt'
with open(path_teams, 'r') as f:
    tmp = f.readlines()
keys = [i.rstrip() for i in tmp]

# load most recent articles
recent_articles = pd.read_pickle('url_titles_newest5.pickle')

# load title information for stories
with open('title_url.pickle', 'rb') as handle:
    title_url = pickle.load(handle)

# build api for y
DEVELOPER_KEY = keys[0]
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)



# # embed_url = '//www.youtube.com/embed/o1BrK2KWifc'
# # embed_url = ['<iframe width="480" height="270" src="//www.youtube.com/embed/Cj_4DupKfuk" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>']
# embed_url = ['<iframe width="480" height="270" src="//www.youtube.com/embed/Cj_4DupKfuk" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>',
#  '<iframe width="480" height="270" src="//www.youtube.com/embed/LgzhCXsBdpA" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>',
#  '<iframe width="480" height="270" src="//www.youtube.com/embed/ui1SquTFnxo" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>']

# pargr = 'Neuroscientist turned data scientist'

#######################
# FUNCTIONS

def get_video_id(q, max_results,token, order="relevance", 
                 location=None, location_radius=None):
    """
    enter in search terms and get out metadate for youtube vido as well as the links to embed videos
    :param q: search terms
    :param max_results: max number of results back you want
    :param token: page search number set to none if want limited returns
    :param order: how to sort output (temporal vs relevance)
    :return: ydict: dictionary with metadata and embedding url
    :return: tok: token for next search
    """
    # get youtube embedding info
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

def get_Table_html(dataframe, recent_articles=None, titles_show=None, max_rows=10, styling=None):
    rows = []
    for i in range(min(len(dataframe), max_rows)):
        row = []
        for col in dataframe.columns:
            if (col == 'Author_wn') or (col == 'Similarity (0-10)'):
                continue
            value = dataframe.iloc[i][col]
            # update this depending on which
            # columns you want to show links for
            # and what you want those links to be
            if col == 'Suggested articles':
                try: 
                    if titles_show != None:
                        title_curr = titles_show[value]
                        if not isinstance(title_curr, str):
                            article = Article(value)
                            article.download()
                            article.parse()
                            title_curr = article.title                            
                    else:
                        article = Article(value)
                        article.download()
                        article.parse()
                        title_curr = article.title
                    cell = html.Td(html.A(href=value, children=title_curr, target='TargetArticle',))
                        # style={'color':'white', 'textDecoration': 'underline'}))
                except:
                    cell = html.Td(children=value)
                    print(value)
            elif col == 'Authors':
                try: 
                    path_link = recent_articles[dataframe.iloc[i]['Author_wn']]['links'][0]
                    # path_link = 'https://muckrack.com/' + dataframe.iloc[i]['Author_wn']
                    cell = html.Td(html.A(href=path_link, children=value, target='TargetArticle',))
                        # style={'color':'white', 'textDecoration': 'underline'}))
                except:
                    cell = html.Td(children=value)
                    print(value)
            else:
                cell = html.Td(children=value)
            row.append(cell)
        rows.append(html.Tr(row))
    return html.Table(
        # Header
        # [html.Tr([html.Th(col) for col in dataframe.columns if (col == 'Author_wn') or (col == 'Similarity (0-10)')])] +

        rows, style=styling
    )

def onLoad_author_names():
    '''Actions to perform upon initial page load'''
    tmpR = writer_features['author_fn'].tolist()
    tmpR.sort()
    author_options = (
        [{'label': generic, 'value': generic}
         for generic in tmpR]
    )
    return author_options

def get_video_rows(embed_url):
    rows = []
    for rr in range(0,1):
        rows.append(html.Iframe(src=embed_url[rr], width='260', height='150'))
    return html.Div([rows]) 

# def generate_about_me_text():
#     return [
#         html.Div([html.P('Neuroscientist turned data scientist with a love for the game', 
#             style={'fontSize': 20})
#             ], style={'position': 'absolute', 'left': -800, 'top': 200})
#     ]


def generate_tables_author_vs_author_similarity(df_authors, recent_articles, max_rows=5):
    return [
    html.Div([
        html.Div([
            html.Br([]),
            html.Br([]),
            html.Br([])]),

        html.H4(children='Nothing but net: click on a name to view his/her most recent article!',
            style={'float': 'inherit', 'color': 'rgb(255,255,255)'
                       }
                ),
        ], className='row'),


    html.Div([
        html.Div([
            html.Div([
                html.Br([])]),

            html.H5(children='''Authors:''' , 
                style={'textAlign': 'left', 'float': 'none', "margin-left": 60, 'color': 'white'} ),

            get_Table_html(df_authors, recent_articles, max_rows=5,
                styling={'float': 'inherit', 'margin-left': 60, 'display': 'block', 
                'padding-left': '10px', 'padding-right': '10px'})

            ], className="two columns"),
        ]),


        html.Div([
            html.Div([
                html.Br([]),
                html.Br([])]),


            html.Div([

                # html.Div([
                #     html.Br([]),
                #     html.Br([]),
                #     html.Br([]),
                #     html.Br([]),
                #     html.Br([]),
                #     html.Br([]),
                #     html.Br([]),
                #     html.Br([]),
                #     html.Br([]),
                #     html.Br([]),
                #     html.Br([]),
                #     html.Br([]),
                #     html.Br([])]),

                html.Iframe(src="", width='1200', height='900', name='TargetArticle',
                    style={'float': 'inherit', 'border': 'none'}),

                ], style={ 'float': 'inherit', 'border': 'none'}),

        ], className="seven columns")


    ]



def generate_tables_author_content_similarity(df_authors, df_content, urls_to_show, recent_articles, 
    title_url, max_rows=5):
	return [
    html.Div([
    html.Div([
        html.Div([
            html.Br([]),
            html.Br([]),
            html.Br([])]),

        html.H4(children='Nothing but net: click on a name, article, or video to view suggested content!',
            style={'float': 'inherit', "color": 'rgb(255,255,255)'
                       }
                ),
        ], className='row'),


	html.Div([
		html.Div([
            html.Div([
                html.Br([])]),

			html.H5(children='''Authors:''' , 
				style={'textAlign': 'left', 'float': 'none', "margin-left": 52, "color": 'white'} ),

            # html.H6(children='''Click on link for most recent article''' , 
            #     style={'textAlign': 'left', 'float': 'none'} ),

			get_Table_html(df_authors, recent_articles, max_rows=5,
                styling={'float': 'inherit', 'margin-left': 52, 'display': 'block', 
                'padding-left': '10px', 'padding-right': '10px'})

			], className="two columns"),


        html.Div([
            html.Div([
                html.Br([])]),

            html.H5(children='''Video:''' , 
                style={'textAlign': 'left', 'float': 'none', 'color': 'white', 'margin-left': 80}),

            html.Iframe(src=urls_to_show[0], width='400', height='200', style={'margin-left': 80}),
            html.Br([]),
            html.Iframe(src=urls_to_show[1], width='400', height='200', style={'margin-left': 80}),
            # html.Br([]),
            # html.Iframe(src=urls_to_show[2], width='300', height='150'),
            ], className="five columns"),


		html.Div([
			html.Div([
				html.Br([])]),

			html.H5(children='''Topics:''' , 
				style={'textAlign': 'left', 'float': 'none', "margin-left": 10, 'color': 'white'}),

			get_Table_html(df_content, titles_show=title_url, max_rows=5,
                styling={'float': 'inherit', 'margin-left': 10, 'display': 'block', 
                'padding-left': '10px', 'padding-right': '10px'})

			], className="four columns"),






		], className="row"),

        html.Div([
            html.Div([
                html.Br([]),
                html.Br([]),
                html.Br([]),
                html.Br([]),
                html.Br([]),
                html.Br([]),
                html.Br([]),
                html.Br([])])
            ], className="row"),

        html.Div([
            html.Div([
                html.Br([]),
                html.Br([])]),


            html.Div([
                html.Br([]),
                html.Br([]),

                # html.H2(children='''Article viewer:''' , 
                #     style={'textAlign': 'left', 'float': 'left'}),

                html.Div([
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Br([]),
                    html.Br([])]),

                html.Iframe(src="", width='1200', height='900', name='TargetArticle',
                    style={'float': 'center', 'border': 'none', 'display': 'block',
                    'margin': '0 auto'}),

                ], style={ 'float': 'center', 'border': 'none'}),

            ], className='row')
        ], style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}, className='row')

	]


####################
app = dash.Dash(__name__)
app.title = 'Full Court Presser'
# app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})


# colors = {
#     'background': '#111111',
#     'text': '#111111'
# }

app.layout = html.Div(children=[
    html.Div([
        html.Div([
            html.Div([
                html.Br([])]),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width= 125, height = 125,
                style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 
                }),
        html.H1(children='Full Court Presser',
            style={'color': 'white','fontSize': 50}),
        html.H3(children='''find your favorite writers.''' , 
            style={ 'color': 'white','textDecoration': 'none'}),

        html.Div([
            html.Br([])]),
        ]),
        # html.Div([
        #     html.Div([
        #         html.Br([])]),
        # html.H1(children='Full Court Presser',
        #     style={'display': 'inline-block', 'float': 'left', 'color': 'white', 
        #     'fontSize': 50,'margin-top': 40}),
        # html.H3(children='''find your favorite writers.''' , 
        #     style={'display': 'inline-block', 'float': 'right', 'color': 'white',
        #      'textDecoration': 'none','margin-top': 45, 'margin-right': 10}),

        # html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width= 125, height = 125,
        #         style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 
        #         }),
        # html.Div([
        #     html.Br([])]),
        # ], className='bgImage'),


    	# html.Div([
    	# html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width= 170, height = 170,
    	# 		style={'float': 'left'}),

    	# 	# style={'verticalAlign': 'middle'}),

    	# # html.Div([
     # #    	html.Br([])]),

     #    html.H1(children='Full Court Presser',
    	#     style={'float': 'inherit', 'color': 'white'
    	#                }
     #        ),

    	# ]),
        # html.H2(children='''find your favorite writers.''' , 
        # 	style={'textAlign': 'left', 'float': 'none', 'color': 'white', 'textDecoration': 'none'}),
        html.Div([html.Br([]),
            html.Br([])]),
        # input bars
        html.Div([
            dcc.Input(id='url-input-box', type='text', size=55, 
                placeholder='Enter url of NBA article'),
            html.P("AND/OR", style={ 'display': 'block', 
                "margin-right": 25, "margin-left": 25}),
            # dcc.Input(id='author-input-box', type='text', size=40, 
            #     placeholder='Enter favorite NBA author',
            #     style= {'horizontalAlign':"left", 'float': 'none'}),
            html.Div([
            html.Div([dcc.Dropdown(id='author-input-box',
                placeholder='Select favorite NBA author',options=onLoad_author_names())],
                style={ "width" : "25%",'color': 'black', 'float': 'center',
                'display': 'block','margin': '0 auto'}),
            html.Button(id='submit-button', n_clicks=0, children='SUBMIT'),]),
            ]),
        # html.Div([html.Br([]),
        #     html.Br([]),
        #     html.Br([]),
        #     html.Br([]),
        #     html.Br([])]),

    ]),
    html.Div(id='output-state'),

    # html.Div([
    #     # html.H4(html.A(href=pargr, children='''About me''', target='TargetArticle', 
    #     #     style={'color': 'black'})),
    #     html.Button(id='my-button', n_clicks=0, children='About me',
    #         style={'color': 'black', 'textTransform': 'none', 
    #         'border': 'none', 'fontSize': 22}),
    #     html.Div(id='output-button')
    #     ], style={'position': 'absolute', 'left': 990, 'top': 145}),
    # html.Div([
    #     html.H4(children='''About me''',
    #             style={ 'color': 'black'}),
    #     ], style={'position': 'absolute', 'left': 1025, 'top': 130,}),





# html.A(href=path_link, children='''About me''', target='TargetArticle')

],
)

# @app.callback(
#     Output('output-button', 'children'), 
#     [Input('my-button', 'n_clicks')])
# def on_click(number_of_times_button_has_clicked):
#     # print('woo')
#     if number_of_times_button_has_clicked > 0:
#         return generate_about_me_text()

# for url
@app.callback(
    Output('output-state', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('url-input-box', 'value'),
    State('author-input-box', 'value')]
)
def update_output_div(n_clicks, input_value1, input_value2):
    # url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', input_value)
    if n_clicks > 0:
        if not input_value2:
            t = time.time()
            article, lem_txt = s.grab_article(input_value1)
            author_sugg = s.give_suggestion_featurespace_single_article(writer_features, txtstr=article.text)
            article_sugg = s.recommend_article_content(kv, w2v_df, lem_text=lem_txt)
            search_string = s.get_search_terms(article.title, team_names)
            if search_string == '': # if empty put as search term the first author
                search_string = article.authors[0]
            print(search_string)
            youtube_ouput, b = get_video_id(search_string, 2, None)
            # urls_to_pass = s.parse_iframe_html(embed_url) 
            urls_to_pass = s.parse_iframe_html(youtube_ouput['embed_url'])
            elapsed = time.time() - t
            print(elapsed)
            return generate_tables_author_content_similarity(author_sugg, article_sugg, urls_to_pass, 
                recent_articles, title_url)
        elif not input_value1:
            author_sugg = s.give_author_suggestion_from_author(writer_features, input_value2)
            print(author_sugg)
            if isinstance(author_sugg, str):
                return author_sugg
            else:
                return generate_tables_author_vs_author_similarity(author_sugg, recent_articles)
                # get_Table_html(author_sugg, recent_articles, max_rows=5)
        else:
            article, lem_txt = s.grab_article(input_value1)
            author_sugg = s.give_author_suggestion_from_author(writer_features, input_value2)
            article_sugg = s.recommend_article_content(kv, w2v_df, lem_text=lem_txt)
            if isinstance(author_sugg, str):
                return author_sugg
            else:
                search_string = s.get_search_terms(article.title, team_names)
                if search_string == '': # if empty put as search term the first author
                    search_string = article.authors[0]
                youtube_ouput, b = get_video_id(search_string, 2, None)
                # urls_to_pass = s.parse_iframe_html(embed_url) 
                urls_to_pass = s.parse_iframe_html(youtube_ouput['embed_url'])
                return generate_tables_author_content_similarity(author_sugg, article_sugg, urls_to_pass, 
                    recent_articles, title_url)


# for new dropdown menus
    # @app.callback(
    #     dash.dependencies.Output('opt-dropdown', 'options'),
    #     [dash.dependencies.Input('name-dropdown', 'value')]
    # )
    # def update_date_dropdown(name):
    #     return [{'label': i, 'value': i} for i in fnameDict[name]]

    # @app.callback(
    #     dash.dependencies.Output('display-selected-values', 'children'),
    #     [dash.dependencies.Input('opt-dropdown', 'value')])
    # def set_display_children(selected_value):
    #     return 'you have selected {} option'.format(selected_value)


# if __name__ == '__main__':
#     app.run_server(debug=True)
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)