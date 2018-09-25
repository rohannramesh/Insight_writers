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
    authors = norm_feature_df['author_list']
    # do similarity test
    similarity_vec = []
    for i in range(0,len(authors)):
        vec1 = norm_feature_df.iloc[i,:-1].values
        result1 = cos_sim(vec1, norm_vec)
        similarity_vec.append(round(result1*10,2)) # multiply bu 10 to scale
    tdf = pd.DataFrame.from_dict({'similarity': similarity_vec, 'authors': authors})
    output_df = tdf.sort_values(by='similarity', ascending=False)
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
        cos_sim_output.append(round(cos_sim(w2v_df[i],article_vector)*10,2)) # multiply by 10 for scale
    r = pd.DataFrame.from_dict({
        'similarity': cos_sim_output, "url": w2v_df.keys()
    })
    df_content_sim = r.sort_values(by='similarity', ascending=False)
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