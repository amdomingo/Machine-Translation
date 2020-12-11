import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from string import digits


#
# file = open('/Users/airadomingo/Documents/Capstone/Code2/bertcebuano-vocab.json')
#
# dict = json.load(file)
#
# dict = pd.DataFrame.from_dict([dict])
#
# dict = dict.transpose()

# LOAD DATA
dir_base_E = "/Users/airadomingo/Documents/Capstone/data/text/text_english.txt"
dir_base_C = "/Users/airadomingo/Documents/Capstone/data/text/text_cebuano.txt"

# FUNCTION TO READ DATA
def read_file(filename):
    '''
    :param filename: text
    :return: opens file
    '''
    input_file_text = open(filename, encoding='latin-1').read()
    return input_file_text


# READ ENGLISH TEXT
eng_txt = read_file(dir_base_E)

# READ CEBUANO TEXT
ceb_txt = read_file(dir_base_C)

# TOKENIZE ENGLISH TEXT
eng_tokens = word_tokenize(eng_txt)

# TOKENIZE CEBUANO TEXT
ceb_tokens = word_tokenize(ceb_txt)

eng_dis = nltk.FreqDist(eng_tokens)
ceb_dis = nltk.FreqDist(ceb_tokens)

eng_dis.plot(20)
ceb_dis.plot(20)

# FUNCTION TO SEPARATE SENTENCE BY SPLITTING AFTER DIGIT
def split_sents(tokens):
    all_sentences = []
    sent = []
    for i in range(len(tokens)-1):
        if tokens[i+1][0].isdigit() != True:
            sent.append(tokens[i])
        else:
            all_sentences.append(sent)
            sent = []

    all_sentences[-1].append(tokens[-1])
    return all_sentences

def clean_text(token_sents):
    '''
    :param token_sents: tokenized sentences
    :return: clean list of lists of words in sentence
    '''
    remove_digits = str.maketrans('', '', digits)
    all_sentences = []
    for i in token_sents:
        sent = []
        for j in i:
            res = j.translate(remove_digits)
            res = res.lower()
            res = res.rstrip()
            sent.append(res)
        all_sentences.append(sent)
    return all_sentences




# SPLIT ENGLISH TEXT INTO SENTENCES
eng_sents = split_sents(eng_tokens)
# print(eng_sents[:50])
print(len(eng_sents))

# CLEAN ENGLISH TEXT
clean_eng = clean_text(eng_sents)
# print(clean_eng[:50])
print(len(clean_eng))

# SPLIT CEBUANO TEXT INTO SENTENCES
ceb_sents = split_sents(ceb_tokens)
print(len(ceb_sents))

# CLEAN CEBUANO TEXT
clean_ceb = clean_text(ceb_sents)
print(len(clean_ceb))


# ===================================
# TOKENIZE ENGLISH TEXT
# clean_eng_tokens = word_tokenize(clean_eng)
#
# # TOKENIZE CEBUANO TEXT
# clean_ceb_tokens = word_tokenize(clean_ceb)

clean_eng_tokens = []
for i in clean_eng:
    for j in i:
        clean_eng_tokens.append(j)
clean_ceb_tokens = []
for i in clean_ceb:
    for j in i:
        clean_ceb_tokens.append(j)


punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''


e_tok =[]
c_tok = []
for i in clean_eng_tokens:
    if i not in punc:
        e_tok.append(i)

for j in clean_ceb_tokens:
    if j not in punc:
        c_tok.append(j)


print(e_tok[:100])

c_eng_dis = nltk.FreqDist(e_tok)
c_ceb_dis = nltk.FreqDist(c_tok)

c_eng_dis.plot(20)
c_ceb_dis.plot(20)

