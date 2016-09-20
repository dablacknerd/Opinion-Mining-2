from nltk import word_tokenize,pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
import numpy as np
import re
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

# tokenize and pos tag
def toke_n_tag(text):
    pos_tagged_text = pos_tag(word_tokenize(text))
    return (pos_tagged_text,text)

def subjectivity_score_1(s):
    return swn.senti_synset(s.name()).obj_score()

def subjectivity_score_2(word,pos):
    neut_arr = []
    if len(swn.senti_synsets(word,pos)) == 0:
        return 0.0
    for s in swn.senti_synsets(word,pos):
        neut_arr.append(s.obj_score())
    #print neut_arr
    return round(np.mean(np.array(neut_arr)),2)
def polarity_score_1(s):
    pos = swn.senti_synset(s.name()).pos_score()
    neg = swn.senti_synset(s.name()).neg_score()
    subj = swn.senti_synset(s.name()).obj_score()
    #print "%s,%s,%s,%s" %(s.name(),pos,neg,subj)
    if pos > neg:
        return pos
    elif neg > pos:
        return neg * -1.0
    elif pos == 0.0 and neg == 0.0:
        return 0.0
    else:
        return 0.0

def polarity_score_2(word,pos):
    pos_arr = []
    neg_arr = []
    neut_arr = []
    if len(swn.senti_synsets(word,pos)) == 0:
        return 0.0
    for s in swn.senti_synsets(word,pos):
        pos_arr.append(s.pos_score())
        neg_arr.append(s.neg_score())
        neut_arr.append(s.obj_score())
    pos = round(np.mean(np.array(pos_arr)),2)
    neg = round(np.mean(np.array(neg_arr)),2) 
    subj = round(np.mean(np.array(neut_arr)),2)
    #return "%s,%s,%s,%s" %(word,pos,neg,subj)
    #print "%s,%s,%s,%s" %(word,pos,neg,subj)
    if pos > neg :
        return pos
    elif neg > pos:
        #return float('-' + str(neg))
        return neg * -1.0
    elif pos == 0.0 and neg == 0.0:
        return 0.0
    else:
        return 0.0

def return_pos_sentiwordnet(pos):
    if pos[:2] == 'NN':
        return 'n'
    elif pos[:2] == 'VB':
        return 'v'
    elif pos[:2] == 'JJ':
        return 'a'
    elif pos[:2] == 'RB':
        return 'r'
    else:
        return 0

def sentiment(text_n_tagged_text):
    pos_tagged_text = text_n_tagged_text[0]
    text = text_n_tagged_text[1]
    pos_arr = []
    neg_arr = []
    subj_arr = []
    obj_arr =[]
    
    for obj in pos_tagged_text:
        if return_pos_sentiwordnet(obj[1]) == 0:
            continue
        pos = return_pos_sentiwordnet(obj[1])
        
        polarity = polarity_score_2(obj[0],pos)
        subj = subjectivity_score_2(obj[0],pos)
        
        if pos == 'n':
            obj_arr.append(subj)
        elif pos == 'a':
            subj_arr.append(subj)
        elif pos == 'v':
            obj_arr.append(subj)
        else:
            subj_arr.append(subj)
        
        if polarity > 0.0:
            pos_arr.append(polarity)
        elif polarity < 0.0:
            neg_arr.append(polarity)
        else:
            continue
    
    if np.array(pos_arr).size == 0:
        pos_mean_score = 0.0
    else:
        pos_mean_score = round(np.mean(np.array(pos_arr)),1)
    if np.array(neg_arr).size == 0:
        neg_mean_score = 0.0
    else:
        neg_mean_score = round(np.mean(np.array(neg_arr)),1)
    
    if len(subj_arr) > len(obj_arr):
        subj_mean_score = round(np.mean(np.array(subj_arr)),1)
    else:
        subj_mean_score = round(np.mean(np.array(obj_arr)),1)
    #subj_mean_score = round(np.mean(np.array(subj_arr)),1)
    if neg_mean_score == 0.0:
        temp_neg_score = neg_mean_score
    else:
        temp_neg_score = neg_mean_score * -1.0
    
    #return {'+ve_polarity':pos_mean_score, '-ve_polarity':neg_mean_score, 'subjectivity':subj_mean_score}
    if pos_mean_score > temp_neg_score:
        return ('1',pos_mean_score + neg_mean_score,subj_mean_score)
    elif pos_mean_score < temp_neg_score:
        return ('-1',pos_mean_score + neg_mean_score,subj_mean_score)
    else:
        return ('0',0.0,subj_mean_score)

def sentiment_wsd(text_n_tagged_text):
    pos_tagged_text = text_n_tagged_text[0]
    text = text_n_tagged_text[1]
    pos_arr = []
    neg_arr = []
    subj_arr = []
    obj_arr = []
    
    for obj in pos_tagged_text:
        if return_pos_sentiwordnet(obj[1]) == 0:
            continue
        pos = return_pos_sentiwordnet(obj[1])
        
        if lesk(text,obj[0],pos):
            syn = lesk(text,obj[0],pos)
            polarity = polarity_score_1(syn)
            subj = subjectivity_score_1(syn)
        else:
            polarity = polarity_score_2(obj[0],pos)
            subj = subjectivity_score_2(obj[0],pos)
        
        if pos == 'n':
            obj_arr.append(subj)
        elif pos == 'a':
            subj_arr.append(subj)
        elif pos == 'v':
            obj_arr.append(subj)
        else:
            subj_arr.append(subj)
        
        if polarity > 0.0:
            pos_arr.append(polarity)
        elif polarity < 0.0:
            neg_arr.append(polarity)
        else:
            continue
        #print pos_arr
    if np.array(pos_arr).size == 0:
        pos_mean_score = 0.0
    else:
        pos_mean_score = round(np.mean(np.array(pos_arr)),1)
    if np.array(neg_arr).size == 0:
        neg_mean_score = 0.0
    else:
        neg_mean_score = round(np.mean(np.array(neg_arr)),1)
    
    if len(subj_arr) > len(obj_arr):
        subj_mean_score = round(np.mean(np.array(subj_arr)),1)
    else:
        subj_mean_score = round(np.mean(np.array(obj_arr)),1)
    if neg_mean_score == 0.0:
        temp_neg_score = neg_mean_score
    else:
        temp_neg_score = neg_mean_score * -1.0
    
    #return {'+ve_polarity':pos_mean_score, '-ve_polarity':neg_mean_score, 'subjectivity':subj_mean_score}
    if pos_mean_score > temp_neg_score:
        return ('1',pos_mean_score + neg_mean_score,subj_mean_score)
    elif pos_mean_score < temp_neg_score:
        return ('-1',pos_mean_score + neg_mean_score,subj_mean_score)
    else:
        return ('0',0.0,subj_mean_score)