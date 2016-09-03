from nltk import word_tokenize,pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
import numpy as np
import re
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

def create_syn_set_annotation(text_pos,text):
    text_syn_set_list =[]
    for t in text_pos:
        pos = None
        if t[1][:2] =='NN':
            pos = 'n'
        elif t[1][:2] =='VB':
            pos ='v'
        elif t[1][:2] =='RB':
            pos = 'r'
        elif t[1][:2] =='JJ':
            pos = 'a'
        
        if lesk(text,t[0],pos):
            text_syn_set_list.append(lesk(text,t[0],pos))
        elif re.match(r'[a-zA-Z]*-[a-zA-Z]*',t[0]) and lesk(text,t[0],pos) is None:
            text_syn_set_set_list.append(wn.synsets(t[0])[0])
        else:
            continue
    return text_syn_set_list

def preprocess_after_postag(text_tokens_pos):
    new_text_pos =[]
    for tp in text_tokens_pos:
        if re.match(r'[a-zA-Z]*-[a-zA-Z]*',tp[0]):
            tp_lisp = tp[0].split('-')
            for tiny in tp_lisp:
                   new_text_pos.append((tiny,tp[1]))
        else:
            new_text_pos.append(tp)
    return new_text_pos

def polarity_score(text_syn_set_list):
    limp =[]
    for s in text_syn_set_list:
        pos_score = swn.senti_synset(s.name()).pos_score()
        neg_score = swn.senti_synset(s.name()).neg_score()
        neut_score = swn.senti_synset(s.name()).obj_score()
        polarity_score = 0
        if pos_score > neg_score and pos_score > neut_score:
            polarity_score = 1
        elif neg_score > pos_score and neg_score > neut_score:
            polarity_score = -1
        else:
            polarity_score = 0
        limp.append(polarity_score)
    arr = np.array(limp)
    return round(arr.mean(),1)

def polarity_score_2(text_syn_set_list):
    limp =[]
    for s in text_syn_set_list:
        pos_score = swn.senti_synset(s.name()).pos_score()
        neg_score = swn.senti_synset(s.name()).neg_score()
        polarity_score = pos_score - neg_score

        limp.append(polarity_score)
    arr = np.array(limp)
    return round(arr.mean(),2)
    
def subjectivity_score(text_syn_set_list):
    limp = []
    for s in text_syn_set_list:
        limp.append(swn.senti_synset(s.name()).obj_score())
    arr = np.array(limp)
    return round(np.mean(arr),1)

def sentiment_analysis(text):
    #lemma = WordNetLemmatizer()
    text_token = word_tokenize(text)
    text_pos = preprocess_after_postag(pos_tag(text_token))
    #text_pos_norm =[]
    #new_text_list =[]
    #for t in text_pos:
        #lem = lemma.lemmatize(t[0])
        #text_pos_norm.append((lem,t[1]))
        #new_text_list.append(lem)
    #new_text = ' '.join(new_text_list)
    syn_set_list = create_syn_set_annotation(text_pos,text)
    polarity = polarity_score_2(syn_set_list)
    subjectivity = subjectivity_score(syn_set_list)
    return (polarity,subjectivity)

def 