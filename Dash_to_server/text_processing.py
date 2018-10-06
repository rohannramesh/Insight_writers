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

class ProcessArticle:
    """
    Class for basic processing of the contents of an article.
    """
    def __init__(self, text):
        self.text = text
        self.cleanedtext = []

    def lemstr(self):
        """
        lemmatize the text but don't save
        :return:
        """
        lz = WordNetLemmatizer()
        newstr = [lz.lemmatize(curr_word, self.get_pos(curr_word)) for curr_word in self.text]
        return newstr
        

    def clean_text(self):
        """
        function to clean the article - ie remove extra lines, spaces, punctuation, and make lowercase
        :return:
        """
        cleanedtext = re.sub(r'\n\s*\n', '', self.text)  # extra lines
        cleanedtext = re.sub(r'[^\w\s]', '', cleanedtext)  # punctuation
        # for numbers
        #     cleanedtext = re.sub(r'[^A-Za-z0-9(),!?@\'\`\"\_\n]', ' ', cleanedtext)
        #     cleanedtext = re.sub(r'\w*\d\w*','',cleanedtext) # words with number plust letters
        #     remove_digits = str.maketrans('', '', digits)
        #     cleanedtext = cleanedtext.translate(remove_digits)
        cleanedtext = re.sub(' +', ' ', cleanedtext)  # extra white spaces
        cleanedtext = cleanedtext.lower()
        self.cleanedtext = cleanedtext

    def get_nwords(self, strcalc=None):
        """
        get the number of words from the cleaned text
        :return:
        """
        if strcalc == None:
            b = word_tokenize(self.cleanedtext)
            n_words = len(b)
        else:
            b = word_tokenize(strcalc)
            n_words = len(b)
        return n_words

    def get_nsentences(self):
        """
        get the number of sentences
        :return: the number of sentences, the number of words per sentence,
        the std of the number of words per sentence
        """
        b = sent_tokenize(self.text)
        n_sentences = len(b)
        nwords = []
        for curr_sent in b:
            nwords.append(self.get_nwords(curr_sent))
        return n_sentences, np.mean(nwords), np.std(nwords)

    def get_word_length(self):
        """
        get information about each word from the cleanedtext
        :return: the average length of the word, the std of length of the word
        the skew of the length of the word
        """
        tokens = word_tokenize(self.cleanedtext)
        # Filter out punctuation
        no_punct_tokens = ([token for token in tokens
                            if any(c.isalpha() for c in token)])
        # Get a distribution of token lengths
        token_lengths = [len(token) for token in no_punct_tokens]
        return np.mean(token_lengths), np.std(token_lengths), skew(token_lengths)

    def get_sentiments_of_sentences(self):
        """
        perform sentiment analysis on each sentence in the article
        :return: the average positive, neutral, and negative sentiment of the article pooled over sentences,
        the variability (std) of this sentiment
        """
        analyzer = SentimentIntensityAnalyzer()
        avg_sentim = []
        sentim_var = []
        sentim_all = {}
        sentim_all['neg'] = []
        sentim_all['neu'] = []
        sentim_all['pos'] = []
        sentim_all['compound'] = []
        sentences = sent_tokenize(self.text)
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

    def build_feature_vector_for_article(self):
        self.clean_text()
        n_words = self.get_nwords()
        n_sentences, n_wordspersentence, n_wordspersent_variability = self.get_nsentences()
        a, b = self.get_sentiments_of_sentences()
        neg_sent = a[0][0]
        neu_sent = a[0][1]
        pos_sent = a[0][2]
        neg_sent_var = b[0][0]
        neu_sent_var = b[0][1]
        pos_sent_var = b[0][2]
        wordlength, wordlength_var, wordlength_skew = self.get_word_length()
        output_vec = [n_words, neg_sent, neu_sent, pos_sent, neg_sent_var,
                      neu_sent_var, pos_sent_var, n_sentences,
                      n_wordspersentence, n_wordspersent_variability,
                      wordlength, wordlength_var, wordlength_skew]
        return output_vec

    def get_pos_article(self):
        nostopwords = self.remove_stopwords(word_tokenize(self.cleanedtext))
        typesofspeech = [self.get_pos(i) for i in nostopwords]
        return [typesofspeech.count('n'), typesofspeech.count('a'), typesofspeech.count('v'), 
            typesofspeech.count('r')]


    @staticmethod
    def get_pos(word):
        """
        get the part of speech of a given word
        :param word:
        :return: part of speech
        """
        w_synsets = wordnet.synsets(word)

        pos_counts = Counter()
        pos_counts["n"] = len([item for item in w_synsets if item.pos() == "n"])
        pos_counts["v"] = len([item for item in w_synsets if item.pos() == "v"])
        pos_counts["a"] = len([item for item in w_synsets if item.pos() == "a"])
        pos_counts["r"] = len([item for item in w_synsets if item.pos() == "r"])

        most_common_pos_list = pos_counts.most_common(3)
        return most_common_pos_list[0][0]  # first indexer for
        # getting the top POS from list, second indexer for getting POS from tuple( POS: count )

    @staticmethod
    def remove_stopwords(word_tokens): # must be tokenized sentence
        """
        remove stopwords like the
        :param word: all word tokens
        :return: word tokent without stop words
        """
        stop_words = set(stopwords.words('english')) 
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return filtered_sentence






