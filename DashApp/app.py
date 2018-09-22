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


	html.Div([html.Br([]),
    	html.Br([])]),

    html.Div([
    	dcc.Input(id='url-input-box', type='text', size=130, 
    		placeholder='Enter url of NBA article', style= {'horizontalAlign':"left", 'float': 'none'}),
    	html.Button(id='submit-button', n_clicks=0, children='Submit'),

        dcc.Input(id='author-input-box', type='text', size=45, 
            placeholder='Enter favorite NBA author', style= {'horizontalAlign':"left", 'float': 'none'}),
        html.Button(id='submit-button2', n_clicks=0, children='Submit'),
        html.Div(id='output-state-author'),
        html.Div(id='output-state'),
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
	article, lem_txt = s.grab_article(input_value)
	author_sugg = s.give_suggestion_featurespace_single_article(writer_features, txtstr=article.text)
	article_sugg = s.recommend_article_content(kv, w2v_df, lem_text=lem_txt)
	return generate_tables_author_content_similarity(author_sugg, article_sugg)

# for author
@app.callback(
    Output('output-state-author', 'children'),
    [Input('submit-button2', 'n_clicks')],
    [State('author-input-box', 'value')]
)
def update_output_div_author(n_clicks, input_value):
    author_sugg = s.give_author_suggestion_from_author(writer_features, input_value)
    return get_Table_html(author_sugg)


if __name__ == '__main__':
    app.run_server(debug=True)