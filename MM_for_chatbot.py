#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:36:20 2019

@author: salihemredevrim, DigitalDigger
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import spacy
import pickle
from nltk.tag import StanfordNERTagger
import string
import re
from replacers import AntonymReplacer
from spellchecker import SpellChecker
import bert_for_chatbot as bert


# %%

##Training data (all safecity and other reviews) 
# data1 = pd.read_excel('MM_last.xlsx')
##data1['processed'] = data1['processed'].astype(str)
# data1['Corpus'] = data1['Corpus'].astype(str)
#
# data1['len1'] = data1.apply(lambda x: len(x.Corpus.strip()) if len(x.Corpus.strip()) > 0 else 0, axis=1)
# data1 = data1[data1['len1'] > 10]
#
##For harassment types
# data2 = data1[data1['Target'] == 1]

# %%
# Functions for classification models
def tfidf(train_data, text, target, min_df, max_df, ngram_range1, ngram_range2):
    # min_df is used for removing terms that appear too infrequently. For example:
    # min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
    # min_df = 5 means "ignore terms that appear in less than 5 documents".
    # max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:
    # max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
    # max_df = 25 means "ignore terms that appear in more than 25 documents".

    # ngram added

    # Tf-Idf ******************************************************************************************************************

    # balance!!
    target1 = train_data[train_data[target] == 1]
    target2 = train_data[train_data[target] == 0]

    min1 = min(len(target1), len(target2))

    data11 = target1.sample(n=min1, random_state=1905)
    data22 = target2.sample(n=min1, random_state=1905)

    train_data2 = data11.append(data22, ignore_index=True)

    # transform
    vect_tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=(ngram_range1, ngram_range2)).fit(
        train_data2[text].astype(str))

    X_train_vectorized_tf = vect_tfidf.transform(train_data2[text].astype(str))

    # target
    YY_train = pd.DataFrame(train_data2[target])

    # Logistic Regression
    model_tfidf = LogisticRegression()

    model_tfidf.fit(X_train_vectorized_tf, YY_train)

    return model_tfidf, vect_tfidf


# Functions for Doc2Vec
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


def dbow(train_data, text, target, vector_size1, window1, negative1, min_count1):
    # DBOW is the Doc2Vec model analogous to Skip-gram model in Word2Vec.
    # The paragraph vectors are obtained by training a neural network on the task of predicting a probability distribution of words in a paragraph given a randomly-sampled word from the paragraph.
    # We set the minimum word count to 2 in order to discard words with very few occurrences.

    # balance!!
    target1 = train_data[train_data[target] == 1]
    target2 = train_data[train_data[target] == 0]

    min1 = min(len(target1), len(target2))

    data11 = target1.sample(n=min1, random_state=1905)
    data22 = target2.sample(n=min1, random_state=1905)

    train_data2 = data11.append(data22, ignore_index=True)

    # transform
    train_doc = pd.DataFrame(pd.concat([train_data2[text].astype(str), train_data2[target]], axis=1))
    train_doc2 = train_doc.apply(lambda x: TaggedDocument(words=tokenize_text(x[text]), tags=[x[target]]), axis=1)

    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=0, vector_size=vector_size1, window=window1, negative=negative1, min_count=min_count1, hs=0,
                         workers=cores, epochs=200)
    train_corpus = [x for x in train_doc2.values]
    model_dbow.build_vocab([x for x in train_doc2.values])

    model_dbow.train(train_corpus, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

    y_train_doc, X_train_doc = get_vectors(model_dbow, train_doc2)

    # Logistic Regression
    model_dbow_log = LogisticRegression()

    model_dbow_log.fit(X_train_doc, y_train_doc)

    return model_dbow, model_dbow_log


# %%
##Models (RUN ONCE!)
##WE CAN CONSIDER ACTIVE LEARNING LATER
#    
# model_tfidf, vect_tfidf = tfidf(data1, 'Corpus', 'Target', 3, 0.95, 1, 3)
# model_tfidf_verbal, vect_tfidf_verbal = tfidf(data2, 'Corpus', 'VERBAL ABUSE', 3, 0.95, 1, 3)
# model_tfidf_nonverbal, vect_tfidf_nonverbal = tfidf(data2, 'Corpus', 'NON-VERBAL ABUSE', 3, 0.95, 1, 3)
# model_tfidf_physical, vect_tfidf_physical = tfidf(data2, 'Corpus', 'PHYSICAL ABUSE', 3, 0.95, 1, 3)
#
# model_dbow, model_dbow_log = dbow(data1, 'Corpus', 'Target', 300, 10, 5, 5)
# model_dbow_verbal, model_dbow_log_verbal = dbow(data2, 'Corpus', 'VERBAL ABUSE', 300, 10, 5, 5)
# model_dbow_nonverbal, model_dbow_log_nonverbal = dbow(data2, 'Corpus', 'NON-VERBAL ABUSE', 300, 10, 5, 5)
# model_dbow_physical, model_dbow_log_physical = dbow(data2, 'Corpus', 'PHYSICAL ABUSE', 300, 10, 5, 5)
#
##Save models 
##TFIDFs
# pickle.dump(model_tfidf, open('model_tfidf.sav', 'wb'))
# with open('vect_tfidf.pk', 'wb') as fin:
#    pickle.dump(vect_tfidf, fin)
#    
# pickle.dump(model_tfidf_verbal, open('model_tfidf_verbal.sav', 'wb'))
# with open('vect_tfidf_verbal.pk', 'wb') as fin:
#    pickle.dump(vect_tfidf_verbal, fin)
#
# pickle.dump(model_tfidf_nonverbal, open('model_tfidf_nonverbal.sav', 'wb'))
# with open('vect_tfidf_nonverbal.pk', 'wb') as fin:
#    pickle.dump(vect_tfidf_nonverbal, fin)
#
# pickle.dump(model_tfidf_physical, open('model_tfidf_physical.sav', 'wb'))
# with open('vect_tfidf_physical.pk', 'wb') as fin:
#    pickle.dump(vect_tfidf_physical, fin)
#
##DOC2VECs
# pickle.dump(model_dbow, open('model_dbow.sav', 'wb'))
# pickle.dump(model_dbow_log, open('model_dbow_log.sav', 'wb'))
#
# pickle.dump(model_dbow_verbal, open('model_dbow_verbal.sav', 'wb'))
# pickle.dump(model_dbow_log_verbal, open('model_dbow_log_verbal.sav', 'wb'))
#
# pickle.dump(model_dbow_nonverbal, open('model_dbow_nonverbal.sav', 'wb'))
# pickle.dump(model_dbow_log_nonverbal, open('model_dbow_log_nonverbal.sav', 'wb'))
#
# pickle.dump(model_dbow_physical, open('model_dbow_physical.sav', 'wb'))
# pickle.dump(model_dbow_log_physical, open('model_dbow_log_physical.sav', 'wb'))

# %%
# CALL MODELS
# models and loading necessary stuff (NO NEED FOR RERUN AT EACH TEXT)

# TFIDF: Harassment or not, types of harassments
model_tfidf = pickle.load(open('./model_tfidf.sav', 'rb'))
vect_tfidf = pickle.load(open('./vect_tfidf.pk', 'rb'))

model_tfidf_verbal = pickle.load(open('./model_tfidf_verbal.sav', 'rb'))
vect_tfidf_verbal = pickle.load(open('./vect_tfidf_verbal.pk', 'rb'))

model_tfidf_nonverbal = pickle.load(open('./model_tfidf_nonverbal.sav', 'rb'))
vect_tfidf_nonverbal = pickle.load(open('./vect_tfidf_nonverbal.pk', 'rb'))

model_tfidf_physical = pickle.load(open('./model_tfidf_physical.sav', 'rb'))
vect_tfidf_physical = pickle.load(open('./vect_tfidf_physical.pk', 'rb'))

# DOC2VEC: harassment or not, types of harasments
model_dbow = pickle.load(open('./model_dbow.sav', 'rb'))
model_dbow_log = pickle.load(open('./model_dbow_log.sav', 'rb'))

model_dbow_verbal = pickle.load(open('./model_dbow_verbal.sav', 'rb'))
model_dbow_log_verbal = pickle.load(open('./model_dbow_log_verbal.sav', 'rb'))

model_dbow_nonverbal = pickle.load(open('./model_dbow_nonverbal.sav', 'rb'))
model_dbow_log_nonverbal = pickle.load(open('./model_dbow_log_nonverbal.sav', 'rb'))

model_dbow_physical = pickle.load(open('./model_dbow_physical.sav', 'rb'))
model_dbow_log_physical = pickle.load(open('./model_dbow_log_physical.sav', 'rb'))

# For Spacy
nlp = spacy.load('en_core_web_md')

# For NER
stanford_ner_tagger = StanfordNERTagger('./stanford_ner/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
                                        './stanford_ner/' + 'stanford-ner-4.0.0.jar')


# %%
# Preprocessing
def decontracted(phrase):
    # specific
    # phrase = phrase.lower() # lowercase text
    phrase = re.sub(r",", "", phrase)
    phrase = re.sub(r'i\'mma', 'i am going to', phrase)
    phrase = re.sub(r'i\'ma', 'i am going to', phrase)
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"ain\'t", "are not", phrase)
    phrase = re.sub(r"gonna", "going to", phrase)
    phrase = re.sub(r"wanna", "want to", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r'dammit', 'damn it', phrase)
    phrase = re.sub(r'imma', 'i am going to', phrase)
    phrase = re.sub(r'gimme', 'give me', phrase)
    phrase = re.sub(r'luv', 'love', phrase)
    phrase = re.sub(r' dem ', 'them', phrase)
    phrase = re.sub(r' asap ', 'as soon as possible', phrase)
    phrase = re.sub(r' gyal ', 'girl', phrase)
    phrase = re.sub(r' dat ', ' that ', phrase)
    phrase = re.sub(r' skrrt ', ' ', phrase)
    phrase = re.sub(r' yea ', ' yeah ', phrase)
    phrase = re.sub(r' ayy ', '', phrase)
    phrase = re.sub(r' aye ', '', phrase)
    phrase = re.sub(r' ohoh ', '', phrase)
    phrase = re.sub(r' hol ', 'hold', phrase)
    phrase = re.sub(r' lil ', ' little ', phrase)
    phrase = re.sub(r' g ', ' gangster ', phrase)
    phrase = re.sub(r' gangsta ', ' gangster ', phrase)
    phrase = re.sub(r'thang', 'thing', phrase)
    phrase = re.sub(r'gotta', 'going to', phrase)
    phrase = re.sub(r' hook ', ' ', phrase)
    phrase = re.sub(r' intro ', ' ', phrase)
    phrase = re.sub(r' gon ', ' going to ', phrase)
    phrase = re.sub(r' shoulda ', ' should have ', phrase)
    phrase = re.sub(r' em ', ' them ', phrase)
    phrase = re.sub(r' ya ', ' you ', phrase)
    phrase = re.sub(r' da ', ' the ', phrase)
    phrase = re.sub(r' na na ', ' ', phrase)
    phrase = re.sub(r' hoe', ' whore', phrase)
    phrase = re.sub(r' oh ', ' ', phrase)
    phrase = re.sub(r'\b(\w+)( \1\b)+', r'\1', phrase)
    phrase = re.sub(r'\'til', 'till', phrase)
    phrase = re.sub(r'ooh', '', phrase)
    phrase = re.sub(r'lala', '', phrase)
    phrase = re.sub(r' ho ', ' whore ', phrase)
    phrase = re.sub(r' mm ', '  ', phrase)
    phrase = re.sub(r' yah ', '  ', phrase)
    phrase = re.sub(r' yeah ', '  ', phrase)
    phrase = re.sub(r'hitta', 'nigga', phrase)
    # phrase = re.sub(r'u', 'you', phrase)
    phrase = re.sub(r'\&', 'and', phrase)
    phrase = re.sub(r'nothin', 'nothing', phrase)
    phrase = re.sub(r'\$', 's', phrase)
    phrase = re.sub(r" c\'mon", "come on", phrase)
    phrase = re.sub(r" \'cause", " because", phrase)
    phrase = re.sub(r" cuz ", " because ", phrase)
    phrase = re.sub(r" \'cuz ", " because ", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\'yo", "your", phrase)

    return phrase


# %%
def spelling11(data1, text1):
    spell = SpellChecker()
    spell.word_frequency.load_text_file('corporaForSpellCorrection.txt')
    sent = data1[text1].str.split()

    for k in range(len(sent)):
        misspelled = spell.unknown(sent.iloc[k])
        xd1 = ''
        for word in sent.iloc[k]:
            if word in misspelled:
                # Get the one `most likely` answer
                word = spell.correction(word)
                xd1 = xd1 + ' ' + word
            else:
                xd1 = xd1 + ' ' + word

        data1[text1].iloc[k] = xd1

    return data1


# %%
def clean_text(texto, min_char, text):
    texto[text] = texto[text].dropna()
    texto[text] = texto[text].astype(str)
    texto[text] = texto[text].replace({'\n': ' '}, regex=True)
    # texto[text] = texto[text].replace(r'[\W_]+', ' ', regex=True)

    remove = string.punctuation
    remove = remove.replace(":", "")
    remove = remove.replace("'", "")

    texto[text] = texto[text].str.translate({ord(char): None for char in remove})

    texto[text] = texto[text].str.lower()  # lowercase text

    # remove weird chars
    texto[text] = texto[text].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

    # texto = pd.DataFrame(texto)
    # keep longer one if description is shorter than 30 chars
    # texto['len1'] = texto.apply(lambda x: len(x.text.strip()) if len(x.text.strip()) > 0 else 0, axis=1)

    # drop if shorter than min_char
    # texto = texto[texto['len1'] >= min_char]

    texto[text] = texto[text].str.strip()
    # texto[text] = texto.dropna()

    # remove weird duplicates
    texto = texto.drop_duplicates()
    texto = texto.reset_index(drop=True)

    return texto


# %%
def negations(text):
    replacer = AntonymReplacer()

    sent = text.split()
    noneg = replacer.replace_negations(sent)
    separator = ' '
    out = separator.join(noneg)

    return out


# %%
# Processed lemmatized version for incoming text
def spacy_data(data1, lyrics):
    corpus = []

    for i in range(0, len(data1)):
        song = data1.iloc[i][lyrics]
        doc = nlp(song)
        spacy_data = pd.DataFrame()

        for token in doc:
            if token.lemma_ == "-PRON-":
                lemma = token.text
            else:
                lemma = token.lemma_
            row = {
                "word": token.text,
                "lemma": lemma,
                "pos": token.pos_,
                "stop_word": token.is_stop
            }
            spacy_data = spacy_data.append(row, ignore_index=True)

        corpus_clean = " ".join(spacy_data["lemma"].values)
        corpus_clean = re.sub(r'[^A-Za-z0-9]+', ' ', corpus_clean)
        corpus.append(corpus_clean)

    data1['Corpus'] = corpus

    return data1


# %%
# Stanford NER
def ner_bert(data1, lyrics):
    return bert.classifyWithBert(data1.iloc[0][lyrics], data1)


def ner_stanford(data1, lyrics):
    print('data1')
    print(data1)
    print('lyrics')
    print(lyrics)

    data1['Location'] = '';
    data1['Date'] = '';
    data1['Time'] = '';

    for i in range(0, len(data1)):

        song = data1.iloc[i][lyrics]
        results = stanford_ner_tagger.tag(song.split())

        for result in results:

            tag_value = result[0]
            tag_type = result[1]

            if tag_type == 'LOCATION':
                data1['Location'].iloc[i] = data1['Location'].iloc[i] + ' ' + tag_value
            elif tag_type == 'DATE':
                data1['Date'].iloc[i] = data1['Date'].iloc[i] + ' ' + tag_value
            elif tag_type == 'TIME':
                data1['Time'].iloc[i] = data1['Time'].iloc[i] + ' ' + tag_value

    return data1


# %%
# FINAL FUNCTION
# takes incoming text, cleans it and returns outputs

def finale(text, min_char_to_run, cut_off):
    # Process text *********************************************************************************************
    clean_text1 = pd.DataFrame(index=range(1))
    clean_text1['text'] = text
    clean_text1['Target'] = -9

    clean_text1['text'] = clean_text1['text'].apply(decontracted)

    # Stanford NER (before lowercase)
    # clean_text1 = ner_stanford(clean_text1, 'text')

    # Bert NER (before lowercase)
    clean_text1 = ner_bert(clean_text1, 'text')

    clean_text1 = clean_text(clean_text1, min_char_to_run, 'text')
    clean_text1['text'] = clean_text1['text'].str.lower()
    clean_text1 = spelling11(clean_text1, 'text')
    clean_text1['text'] = clean_text1['text'].apply(negations)

    # POS
    clean_text1 = spacy_data(clean_text1, 'text')
    # clean_text1['Corpus'] = clean_text1['Corpus'].lower()

    # manual check for length (for pos)!
    clean_text1['len1'] = clean_text1.apply(lambda x: len(x.Corpus.strip()) if len(x.Corpus.strip()) > 0 else 0, axis=1)

    if clean_text1['len1'].item() > min_char_to_run:

        # Predictions **************************************************************************************************
        # TFIDFs
        test_vectorized_1 = vect_tfidf.transform(clean_text1['Corpus'])
        clean_text1['tfidf_1'] = model_tfidf.predict_proba(test_vectorized_1)[:, 1]
        clean_text1['tfidf_1'] = clean_text1['tfidf_1'].apply(lambda x: 1 if x >= cut_off else 0)

        test_vectorized_2 = vect_tfidf_verbal.transform(clean_text1['Corpus'])
        clean_text1['tfidf_v'] = model_tfidf_verbal.predict_proba(test_vectorized_2)[:, 1]
        clean_text1['tfidf_v'] = clean_text1['tfidf_v'].apply(lambda x: 1 if x >= cut_off else 0)

        test_vectorized_3 = vect_tfidf_nonverbal.transform(clean_text1['Corpus'])
        clean_text1['tfidf_nv'] = model_tfidf_nonverbal.predict_proba(test_vectorized_3)[:, 1]
        clean_text1['tfidf_nv'] = clean_text1['tfidf_nv'].apply(lambda x: 1 if x >= cut_off else 0)

        test_vectorized_4 = vect_tfidf_physical.transform(clean_text1['Corpus'])
        clean_text1['tfidf_p'] = model_tfidf_physical.predict_proba(test_vectorized_4)[:, 1]
        clean_text1['tfidf_p'] = clean_text1['tfidf_p'].apply(lambda x: 1 if x >= cut_off else 0)

        # Doc2Vec
        test_doc = pd.DataFrame(pd.concat([clean_text1['Corpus'], clean_text1['Target']], axis=1))
        test_doc2 = test_doc.apply(lambda x: TaggedDocument(words=tokenize_text(x['Corpus']), tags=[x['Target']]),
                                   axis=1)

        y_test_doc_1, X_test_doc_1 = get_vectors(model_dbow, test_doc2)
        clean_text1['dbow_1'] = model_dbow_log.predict_proba(X_test_doc_1)[:, 1]
        clean_text1['dbow_1'] = clean_text1['dbow_1'].apply(lambda x: 1 if x >= cut_off else 0)

        y_test_doc_v, X_test_doc_v = get_vectors(model_dbow_verbal, test_doc2)
        clean_text1['dbow_v'] = model_dbow_log_verbal.predict_proba(X_test_doc_v)[:, 1]
        clean_text1['dbow_v'] = clean_text1['dbow_v'].apply(lambda x: 1 if x >= cut_off else 0)

        y_test_doc_nv, X_test_doc_nv = get_vectors(model_dbow_nonverbal, test_doc2)
        clean_text1['dbow_nv'] = model_dbow_log_nonverbal.predict_proba(X_test_doc_nv)[:, 1]
        clean_text1['dbow_nv'] = clean_text1['dbow_nv'].apply(lambda x: 1 if x >= cut_off else 0)

        y_test_doc_p, X_test_doc_p = get_vectors(model_dbow_physical, test_doc2)
        clean_text1['dbow_p'] = model_dbow_log_physical.predict_proba(X_test_doc_p)[:, 1]
        clean_text1['dbow_p'] = clean_text1['dbow_p'].apply(lambda x: 1 if x >= cut_off else 0)

        # Flags
        clean_text1['Harassment_flg'] = clean_text1['tfidf_1'] + clean_text1['dbow_1']
        clean_text1['Verbal_flg'] = clean_text1['tfidf_v'] + clean_text1['dbow_v']
        clean_text1['NonVerbal_flg'] = clean_text1['tfidf_nv'] + clean_text1['dbow_nv']
        clean_text1['Physical_flg'] = clean_text1['tfidf_p'] + clean_text1['dbow_p']

        count1 = clean_text1['Verbal_flg'] + clean_text1['NonVerbal_flg'] + clean_text1['Physical_flg']

        # Suspicous harassments
        if count1.item() < 2:
            clean_text1['Harassment_flg'] = -9

        clean_text1['text'] = text
        keep_list = ['text', 'Harassment_flg', 'Verbal_flg', 'NonVerbal_flg', 'Physical_flg', 'Location', 'Date',
                     'Time']
        clean_text1 = clean_text1[keep_list]

    else:
        clean_text1['text'] = text
        clean_text1 = pd.DataFrame(clean_text1['text'])
        clean_text1['Harassment_flg'] = -9
        clean_text1['Verbal_flg'] = -9
        clean_text1['NonVerbal_flg'] = -9
        clean_text1['Physical_flg'] = -9
        clean_text1['Location'] = ''
        clean_text1['Date'] = ''
        clean_text1['Time'] = ''

    return clean_text1
