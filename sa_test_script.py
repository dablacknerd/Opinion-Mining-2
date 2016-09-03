import pandas as pd
from tomOpinionMining2 import *

def polarity_mapper(score):
    if score < 0.0:
        return '-1'
    elif score > 0.0:
        return '1'
    else:
        return '0'

def polarity_mapper_1(score):
    if str(score) == '1':
        return '1'
    elif str(score) == '0':
        return '-1'
    else:
        return '0'

file_path = "C:\\Users\\aoguntuga\\Documents\\Datasets\\Sentiment Analysis\\sentiment labelled sentences\\imdb_labelled.txt"
sent_frame = pd.read_csv(file_path,sep='\t',header=None)
sent_frame.columns =['Sentence','Dataset Polarity']

link = list(sent_frame.index.values)
my_polarity_list =[]
my_subjectivity_list =[]

for k in link:
    tex = sent_frame['Sentence'][k].rstrip().decode('utf-8')
    #print tex
    #print k
    score = sentiment_analysis(tex)
    my_polarity_list.append(score[0])
    my_subjectivity_list.append(score[1])

sent_frame['My Polarity Score'] = my_polarity_list
sent_frame['Subjectivity'] = my_subjectivity_list


mapped_pol_1 =[polarity_mapper(x) for x in sent_frame['My Polarity Score']]
sent_frame['My Polarity Score Conv'] = mapped_pol_1
sent_frame['Dataset Polarity Score Conv'] = [polarity_mapper_1(k) for k in sent_frame['Dataset Polarity']]

vin = []
for i in link:
    if sent_frame['My Polarity Score Conv'][i] == sent_frame['Dataset Polarity Score Conv'][i]:
        vin.append(1)
    else:
        vin.append(0)
sent_frame['Polarity Score'] = vin
x = float(sent_frame['Polarity Score'].sum())
total = len(link)
print "Number of matches: %s" % x

x_percent = x/total * 100.0

print "Percentage match: %s percent" % x_percent
out_file ="C:\\Users\\aoguntuga\\Documents\\Sentiment Analysis Research\\Results\\sw_test_3.xlsx"
sent_frame.to_excel(out_file)
print "Done"