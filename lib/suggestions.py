import sys
sys.path.append("/Users/rohanramesh/Documents/GitHub/Insight_writers/lib/")
from text_processing import ProcessArticle as pa
import pandas as pd
from newspaper import Article
from string import digits
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import skew
import re
import numpy as np
from nltk.corpus import wordnet # To get words in dictionary with their parts of speech
from nltk.stem import WordNetLemmatizer # lemmatizes word based on it's parts of speech
from collections import Counter
from nltk.corpus import stopwords 
import gensim
from numpy import dot
from numpy.linalg import norm
from scipy.stats import skew
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from apiclient.discovery import build
from apiclient.errors import HttpError
import difflib
import time


def lemstr(txtstr):
    """
    Lemmatize text string
    :param writer_feature_subsection: df that has each row for an author and each column a feature
    :param author: author name
    :return:
    """
    lz = WordNetLemmatizer()
    newstr = [lz.lemmatize(curr_word, pa.get_pos(curr_word)) for curr_word in txtstr]
    return newstr

def grab_article(article_url):
    """
    grab article from url entered
    """
    article = Article(article_url)
    article.download()
    article.parse()
    lem_art = lemstr(gensim.utils.simple_preprocess(article.text))
    return article, lem_art

def give_author_suggestion_from_author(writer_feature_subsection, author):
    """
    Given an author found in our database, give suggestions for other similar authors
    based on the features we have calculated to define writer style
    :param writer_feature_subsection: df that has each row for an author and each column a feature
    :param author: author name
    :return:
    """
    norm_feature_df = normalize_df(writer_feature_subsection)
    authors = norm_feature_df['author_fn'].tolist()
    author_wn = norm_feature_df['author_list']
    if author not in authors:
        return 'Author not found in database'
    else:
        idx = authors.index(author)
        norm_vec = norm_feature_df.iloc[idx][:-2]
        # do similarity test
        similarity_vec = []
        for i in range(0,len(authors)):
            vec1 = norm_feature_df.iloc[i,:-2].values
            result1 = cos_sim(vec1, norm_vec)
            similarity_vec.append(round(result1*10,1)) # multiply bu 10 to scale
        tdf = pd.DataFrame.from_dict({'Similarity (0-10)': similarity_vec, 
            'Authors': authors, 'Author_wn': author_wn})
        output_df = tdf.sort_values(by='Similarity (0-10)', ascending=False)
        # filter our current author
        output_df = output_df[output_df['Authors'] != author]
        return output_df.iloc[0:10]


def give_suggestion_featurespace_single_article(writer_feature_subsection, txtstr=None, url=None):
    """
    Given a url from an article suggest other similar authors by calculating the features for that
    article and then comparing to the author features
    :param writer_feature_subsection: writer feature pandas dataframe
    :param txtstr: article can be passed directly
    :param url: url for a website
    :return:
    """
    if txtstr is not None:
        curr = pa(txtstr)
        vec = curr.build_feature_vector_for_article()
    else:
        arti = grab_article(url)
        curr = pa(arti)
        vec = curr.build_feature_vector_for_article()
    norm_vec = normalize_vec(vec, writer_feature_subsection.mean().tolist(),
                             writer_feature_subsection.std().tolist())
    norm_feature_df = normalize_df(writer_feature_subsection)
    authors = norm_feature_df['author_fn']
    # website names
    author_wn = norm_feature_df['author_list']
    # do similarity test
    similarity_vec = []
    for i in range(0,len(authors)):
        vec1 = norm_feature_df.iloc[i,:-2].values
        result1 = cos_sim(vec1, norm_vec)
        similarity_vec.append(round(result1*10,1)) # multiply bu 10 to scale
    tdf = pd.DataFrame.from_dict({'Similarity (0-10)': similarity_vec, 'Authors': authors,
         'Author_wn': author_wn})
    output_df = tdf.sort_values(by='Similarity (0-10)', ascending=False)
    # output_df = output_df[output_df['authors'] != author]
    return output_df.iloc[0:10]


def recommend_article_content(keyedvectors, w2v_df, lem_text=None, article_url=None):
    """
    Given a url from an article suggest other articles with similar content due to w2v
    :param keyedvectors: low dimensional vectors from word2vec model
    :param w2v_df: the vectors from previous articles for cos_sim
    :param url: url for a website
    :return: df with recommendations
    """
    if article_url is not None:
        arti, lem_text = grab_article(url)
    # preprocess data
    # a = grab_article_for_w2v(article_url)
    # get vector
    article_vector = get_vector_from_w2v_model(keyedvectors, lem_text)
    # check against current df of articles using cos_sim
    cos_sim_output = []
    for i in w2v_df:
        cos_sim_output.append(round(cos_sim(w2v_df[i],article_vector)*10,1)) # multiply by 10 for scale

    r = pd.DataFrame.from_dict({
        'Similarity (0-10)': cos_sim_output, "Suggested articles": w2v_df.keys()
    })
    df_content_sim = r.sort_values(by='Similarity (0-10)', ascending=False)
    df_content_sim = df_content_sim[df_content_sim['Similarity (0-10)'] != 10]
    return df_content_sim


def get_vector_from_w2v_model(keyedvectors, txtstr):
    """
    Given a url from an article suggest other articles with similar content due to w2v
    :param keyedvectors: low dimensional vectors from word2vec model
    :param txtsr: text from article
    :return: vector representation of article
    """
    a = [r for r in txtstr if r in keyedvectors.vocab]
    array_for_art = np.ndarray(shape=[len(a),keyedvectors.vector_size])
    for i in range(0,len(a)):
        array_for_art[i,:] = keyedvectors[a[i]]
    article_vector = np.mean(array_for_art, axis=0)
    return article_vector


def match_author_names(writer_df, author):
    """
    Reverse website name to full name and vice versa
    :param writer_df: df that has both names
    :param author: writer's name
    :return: the translated name
    """
    wa = writer_df['website_name'].tolist()
    al = writer_df['Idea Text'].tolist()
    if author in wa:
        return al[wa.index(author)]
    elif author in al:
        return wa[al.index(author)]
    else:
        print('improper author name')


def normalize_df(df):
    """
    normalize a df
    :param df: df to be normalized by columns
    :return: normalized df
    """
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
    """
    normalize a list
    :param list1: list to be normalized
    :param means: vector of means
    :param var: vector of std
    :return:
    """
    output_vec = [(list1[i]-means[i])/var[i] for i in range(0,len(list1))]
    return output_vec

def cos_sim(vec1,vec2):
    """
    Calculate cosine similarity between 2 vectors
    """
    a = dot(vec1, vec2)/(norm(vec1)*norm(vec2))
    return a

def get_search_terms(string=None, team_names=None):
    """
    get the input for the search bar for the youtube api based on the named entities in the title of article
    :param string: this is the string of the title
    :param team_names: if you provide a list of team names will also check that in head to head
    :param var: vector of std
    :return:    
    """
    doc = nlp(string)
    a = [i.text for i in doc.ents if (i.label_ == 'PERSON') | (i.label_ == 'GPE')]
    # iterate through each word in string and compare to team_names if any hits then keep
    if team_names is not None:
        tokens = word_tokenize(string)
        for i in tokens:
            if not difflib.get_close_matches(i,team_names, n=1):
                continue
            else:
                m = difflib.get_close_matches(i,team_names, n=1)[0]
            a.append(m)
    # now turn to normal string
    input_search = ['']
    for i in a:
        input_search[0] = input_search[0] + i + ' '
    return input_search[0]

def parse_iframe_html(input_vec):
    """
    Parsing the url for dash iFrame format
    """
    output_url = []
    for i in input_vec:
        idx = i.index('src="')
        tmpR = i[idx+5:]
        idx2 = tmpR.index('"')
        output_url.append(tmpR[:idx2])
    return output_url