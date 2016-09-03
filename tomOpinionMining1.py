from nltk import word_tokenize,pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
import numpy as np
import re
from nltk.corpus import sentiwordnet as swn

def create_syn_set_annotation(text_pos_normalized,text):
    text_syn_set_list =[]
    for t in text_pos_normalized:
        if t[1][:2] =='NN':
            if lesk(text,t[0],'n'):
                text_syn_set_list.append(lesk(text,t[0],'n'))
        elif t[1][:2] =='VB':
            if lesk(text,t[0],'v'):
                text_syn_set_list.append(lesk(text,t[0],'v'))
        elif t[1][:2] =='RB':
            if lesk(text,t[0],'r'):
                text_syn_set_list.append(lesk(text,t[0],'r'))
        elif t[1][:2] =='JJ':
            if lesk(text,t[0],'a'):
                text_syn_set_list.append(lesk(text,t[0],'a'))
        else:
            continue
    return text_syn_set_list

-

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

def sentiment_analysis(text):
    lemma = WordNetLemmatizer()
    text_token = word_tokenize(text)
    text_pos = preprocess_after_postag(pos_tag(text_token))
    text_pos_norm =[]
    new_text_list =[]
    for t in text_pos:
        lem = lemma.lemmatize(t[0])
        text_pos_norm.append((lem,t[1]))
        new_text_list.append(lem)
    new_text = ' '.join(new_text_list)
    syn_set_list = create_syn_set_annotation(text_pos_norm,new_text)
    polarity = polarity_score_2(syn_set_list)
    subjectivity = subjectivity_score(syn_set_list)
    return (polarity,subjectivity)