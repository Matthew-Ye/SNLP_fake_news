import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
import pandas as pd 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

_wnl = nltk.WordNetLemmatizer()


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)

def gen_or_load_feats_with_IDs(feat_fn, headlines, bodies, IDs, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies, IDs)
        np.save(feature_file, feats)

    return np.load(feature_file)

def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X

# def polarity_features_train(headlines, bodies, IDs):
#     train = pd.read_csv('features/train_sentimentByTransformer_trainedOnMoviereview.csv')
#     X = []
#     for i, (headline, body, ID) in tqdm(enumerate(zip(headlines, bodies, IDs))):
#         selected = train[(train['Body ID']==ID) & (train['Headline']==headline)]
#         features = []
#         features.append(int(list(selected['polarity_a'])[0]))
#         features.append(int(list(selected['polarity_b'])[0]))
#         X.append(features)
#     return np.array(X)

# def polarity_features_competition(headlines, bodies, IDs):
#     test = pd.read_csv('features/test_sentimentByTransformer_trainedOnMoviereview.csv')
#     X = []
#     for i, (headline, body, ID) in tqdm(enumerate(zip(headlines, bodies, IDs))):
#         selected = test[(test['Body ID']==ID) & (test['Headline']==headline)]
#         features = []
#         features.append(int(list(selected['polarity_a'])[0]))
#         features.append(int(list(selected['polarity_b'])[0]))
#         X.append(features)
#     return np.array(X)

def keywords_features_competition(headlines, bodies, IDs):
    test = pd.read_csv('features/competition_test_bodies_topics.csv')
    X = []
    for i, (headline, body, ID) in tqdm(enumerate(zip(headlines, bodies, IDs))):
        keywords= list(test[test['Body ID']==ID]['keywords'])[0]
        features = []
        h_ngrams = ngrams_1_2(headline, 2)
        count = 0
        for two_gram in h_ngrams:
            if two_gram in keywords:
                count += 1
        ratio = count / len(h_ngrams)
        if ratio > 0:
            features.append(1)
        else:
            features.append(0)
        features.append(ratio)
        X.append(features)
    return np.array(X)

def keywords_features_train(headlines, bodies, IDs):
    train = pd.read_csv('features/train_bodies_topics.csv')
    X = []
    for i, (headline, body, ID) in tqdm(enumerate(zip(headlines, bodies, IDs))):
        keywords= list(train[train['Body ID']==ID]['keywords'])[0]
        features = []
        h_ngrams = ngrams_1_2(headline, 2)
        count = 0
        for two_gram in h_ngrams:
            if two_gram in keywords:
                count += 1
        ratio = count / len(h_ngrams)
        if ratio > 0:
            features.append(1)
        else:
            features.append(0)
        features.append(ratio)
        X.append(features)
    return np.array(X)

# Calculate polarity / sentiment by VADER
def polarity_features_NLTK(headlines, bodies):

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        passage = ' '.join(tokens)
        return list(sia.polarity_scores(passage).values())
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features += calculate_polarity(clean_headline)
        features += calculate_polarity(clean_body)
        X.append(features)
    return np.array(X)

# Calculate polarity / sentiment by keywords
def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)

# Regard the whole input sequence as one list of different tokens
def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

# Calulate both unigrams and bigrams
def ngrams_1_2(input, n=2):
    input = input.lower().split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(' '.join(input[i:i + n]))
    output += input
    return output

# Regard the whole input sequence as one big string
def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits/len(grams))
    features.append(grams_early_hits/len(grams))
    features.append(grams_first_hits/len(grams))
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits/len(grams))
    features.append(grams_early_hits/len(grams))
    return features


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))


    return X
