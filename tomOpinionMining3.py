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

# perform analysis current
def sentiment_analysis(text_n_tagged_text):
    pos_tagged_text = text_n_tagged_text[0]
    text = text_n_tagged_text[1]
    pos_arr = []
    neg_arr = []
    subj_arr = []
    
    for obj in pos_tagged_text:
        if return_pos_sentiwordnet(obj[1]) == 0:
            continue
        pos = return_pos_sentiwordnet(obj[1])
        
        polarity = polarity_score_2(obj[0],pos)
        subj = subjectivity_score_2(obj[0],pos)
        
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
    subj_mean_score = round(np.mean(np.array(subj_arr)),1)
    temp_neg_score = neg_mean_score * -1.0
        
    #if (pos_mean_score,neg_mean_score,subj_mean_score):
        #return (pos_mean_score,neg_mean_score,subj_mean_score)
    #else:
        #return (0,0.0)
    if pos_mean_score > temp_neg_score:
        return ('1',pos_mean_score + neg_mean_score,subj_mean_score)
    elif pos_mean_score < temp_neg_score:
        return ('-1',pos_mean_score + neg_mean_score,subj_mean_score)
    else:
        return ('0',0.0,subj_mean_score)

# perform analysis with WSD
def sentiment_analysis_wsd(text_n_tagged_text):
    pos_tagged_text = text_n_tagged_text[0]
    text = text_n_tagged_text[1]
    pos_arr = []
    neg_arr = []
    subj_arr = []
    
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
    subj_mean_score = round(np.mean(np.array(subj_arr)),1)
    temp_neg_score = neg_mean_score * -1.0
        
    #if (pos_mean_score,neg_mean_score,subj_mean_score):
        #return (pos_mean_score,neg_mean_score,subj_mean_score)
    #else:
        #return (0,0.0)
    if pos_mean_score > temp_neg_score:
        return ('1',pos_mean_score + neg_mean_score,subj_mean_score)
    elif pos_mean_score < temp_neg_score:
        return ('-1',pos_mean_score + neg_mean_score,subj_mean_score)
    else:
        return ('0',0.0,subj_mean_score)

# perform analysis old
def sentiment_analysis_old(text_n_tagged_text):
    lemma = WordNetLemmatizer()
    pos_tagged_text = text_n_tagged_text[0]
    text = text_n_tagged_text[1]
    n_text = new_sent_after_lemma(text)
    pos_arr = []
    neg_arr = []
    subj_arr = []
    
    polarity_word_arr =[]
    subj_word_arr =[]
    for obj in pos_tagged_text:
        #lem = lemma.lemmatize(obj[0])
        if return_pos_sentiwordnet(obj[1]) == 0:
            continue
        pos = return_pos_sentiwordnet(obj[1])
        if lesk(text,obj[0],pos):
        #if lesk(n_text,lem,pos):
            #syn = lesk(text,obj[0],pos)
            #syn = lesk(n_text,lem,pos)
            #polarity = polarity_score_1(syn)
            polarity = polarity_score_2(obj[0],pos)
            #polarity = polarity_score_2(lem,pos)
            #subj = subjectivity_score_1(syn)
            subj = subjectivity_score_2(obj[0],pos)
            #subj = subjectivity_score_2(lem,pos)
        elif re.match(r'[a-zA-Z]*-[a-zA-Z]*',obj[0]):
            polarity = polarity_score_2(obj[0],pos)
            subj = subjectivity_score_2(obj[0],pos)
        else:
            polarity = polarity_score_2(obj[0],pos)
            subj = subjectivity_score_2(obj[0],pos)
            #continue
        
        subj_arr.append(subj)
        polarity_word_arr.append((obj[0],polarity))
        subj_word_arr.append((obj[0],subj))
        if polarity > 0.0:
            pos_arr.append(polarity)
        elif polarity < 0.0:
            neg_arr.append(polarity)
        else:
            continue
    
    #print subj_word_arr
    pos_mean_score = round(np.mean(np.array(pos_arr)),2)
    neg_mean_score = round(np.mean(np.array(neg_arr)),2)
    subj_mean_score = round(np.mean(np.array(subj_arr)),2)
    scores = {'pos':pos_mean_score,'neg':neg_mean_score,'subj':subj_mean_score}
    temp_neg_score = neg_mean_score * -1.0
    
    if pos_mean_score > temp_neg_score:
        return ('1',pos_mean_score + neg_mean_score,subj_mean_score)
    elif pos_mean_score < temp_neg_score:
        return ('-1',pos_mean_score + neg_mean_score,subj_mean_score)
    else:
        return ('0',0.0,subj_mean_score)
        

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

def new_sent_after_lemma(text):
    lemma = WordNetLemmatizer()
    new_text = []
    for word in pos_tag(word_tokenize(text)):
        pos = return_pos_sentiwordnet(word[1])
        if pos:
            new_w = lemma.lemmatize(word[0],pos = pos ) 
            new_text.append(new_w)
        else:
            new_w = word[0] 
            new_text.append(new_w)

    return ' '.join(new_text)