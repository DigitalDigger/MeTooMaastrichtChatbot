"""
Created on Sat May 11 21:06:03 2019

@author: DigitalDigger
"""

from deeppavlov import configs, build_model
#from jpype import *
from nltk import sent_tokenize
#from natty import DateParser
from collections import Counter
import copy
import requests
import numpy as np


#ner_model = build_model(configs.ner.ner_ontonotes_bert, download=False)
ner_model = build_model(configs.ner.ner_conll2003_bert, download=False)

def combineNERTuples(itemlist, extractedNERs):
    for sentIdx in range(len(itemlist[0])):
        for wordIdx in range(len(itemlist[0][sentIdx])):
            if itemlist[1][sentIdx][wordIdx] != 'O':
                wordNER = [itemlist[0][sentIdx][wordIdx], itemlist[1][sentIdx][wordIdx]]
                extractedNERs.append(wordNER)
    return extractedNERs

def getMergedNERs(extractedNERs):
    # combine NERs in accordance with BIO format
    combinedNERs = []
    nerIdx = 0
    while nerIdx < len(extractedNERs):
        ner = copy.deepcopy(extractedNERs[nerIdx])
        nextNer = nerIdx + 1
        while nextNer < len(extractedNERs) and extractedNERs[nextNer][1][0] == 'I':
            ner = [ner[0] + ' ' + str(extractedNERs[nextNer][0]), ner[1]]
            nextNer = nextNer + 1
            nerIdx = nerIdx + 1
        combinedNERs.append(ner)
        nerIdx = nerIdx + 1

    return combinedNERs

def removeIdenticalNERs(combinedNERs):
    # remove less frequent NERs that have identical names (e.g., Artyom, PERSON and Artyom, GPE)
    combinedNERs = [tuple(l) for l in combinedNERs]
    counts = Counter(combinedNERs)
    mostCommon = copy.deepcopy(counts.most_common())
    print(mostCommon)
    nerIdx = 0
    while nerIdx < len(mostCommon) - 1:
        nerLessFrequentIdx = nerIdx + 1
        while nerLessFrequentIdx < len(mostCommon):
            if mostCommon[nerIdx][0][0] == mostCommon[nerLessFrequentIdx][0][0] \
                    and mostCommon[nerIdx][1] > mostCommon[nerLessFrequentIdx][1]:
                del mostCommon[nerLessFrequentIdx]
                nerLessFrequentIdx -= 1
            nerLessFrequentIdx += 1
        nerIdx += 1
    print(mostCommon)

    return mostCommon

locationsCache = []
nonLocationsCache = []
def checkWikiData(ner):
    if (ner[0] not in locationsCache) and (ner[0] not in nonLocationsCache):
        url = 'https://query.wikidata.org/sparql'
        # ner_alternatives = edits1(ner[0])
        ner_alternatives = [ner[0]]
        print(ner_alternatives)

        for curNer in ner_alternatives:
            query = 'SELECT ?coordinate_location WHERE { ?place rdfs:label \"' + ner[0] + '\"@en . ?place wdt:P625 ?coordinate_location . } LIMIT 1'
            print(query)
            r = requests.get(url, params={'format': 'json', 'query': query})
            # print(ner)
            print(r)

            if r.status_code == 200:
                data = r.json()
                # for item in data['results']['bindings']:
                #     print(item)
                if len(data['results']['bindings']) > 0:
                    print(data['results']['bindings'])
                    locationsCache.append(curNer)
                    ner[1] = 'B-GPE'
                    return data['results']['bindings']
                else:
                    nonLocationsCache.append(curNer)
    elif ner[0] in locationsCache:
        ner[1] = 'B-GPE'

    return None

def classifyWithBert(fileContents, data1):
    #startJVM(getDefaultJVMPath(), "-ea")
    #java.lang.System.out.println("hello world")
    #shutdownJVM()
    sents = sent_tokenize(fileContents)
    sents = ner_model(sents)
    extractedNERs = []
    extractedNERs = combineNERTuples(sents, extractedNERs)
    print('extractedNERs:')
    print(extractedNERs)
    print('Classifying and Writing to file')
    mergedNERs = getMergedNERs(extractedNERs)
    print(mergedNERs)
    # relabel wrong NERs
    for ner in mergedNERs:
        if ner[1] == 'B-PERSON' or ner[1] == 'B-FAC' or ner[1] == 'B-ORG' or ner[1] == 'B-GPE' or ner[1] == 'B-NORP':
            checkWikiData(ner)
#    mostCommon = removeIdenticalNERs(mergedNERs)
#    print(mostCommon)
    
    data1['Location'] = ''; 
    data1['Date'] = '';
    data1['Time'] = '';
    
    for ner in mergedNERs:
        print(ner)
        if ner[1] == 'B-FAC' or ner[1] == 'B-ORG' or ner[1] == 'B-GPE' or ner[1] == 'B-NORP' or ner[1] == 'B-LOC':
            print(ner)
            data1['Location'] = ner[0]
        elif ner[1] == 'B-DATE':
            print(ner)
            #dp = DateParser(ner[0])
            #print(dp.result)
            data1['Date'] = ner[0]
        elif ner[1] == 'B-TIME':
            data1['Time'] = ner[0]
            print(ner)
            #dp = DateParser(ner[0])
            #print(dp.result)
            
    print('Returning to chatbot:')
    print(data1)
    return data1  




