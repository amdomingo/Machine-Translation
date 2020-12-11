# IMPORT LIBRARIES
import os, sys
import numpy as np
import nltk
# nltk.download('punkt')
from string import digits
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#===============================================================

# SET PARAMETERS
BATCH_SIZE = 64
EPOCHS = 30
LSTM_NODES = 256
NUM_SENTENCES = 10000
MAX_SENTENCE_LENGTH = 50
MAX_NUM_WORDS = 20000
EMBEDDING_SIZE = 100

#===============================================================

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

# for token in clean_eng:
#     ' '.join(token)



def join_sents(text):
    sents = []
    for i in text:
        res = ' '.join(i)
        sents.append(res)
    return sents

join_eng = join_sents(clean_eng)

join_ceb = join_sents(clean_ceb)



# FUNCTION TO ADD <SOS> AND <EOS> TAGS
def tagger(input, output, NUM_SENTENCES):
    sos = '<sos>'
    eos = '<eos>'
    input_sentences = []
    output_sentences = []
    output_sentences_inputs = []

    count = 0

    for o in output:
        count += 1
        if count > NUM_SENTENCES:
            break
        else:
            output_sentence = o + ' ' + eos
            output_sentence_input = sos + o

            output_sentences.append(output_sentence)
            output_sentences_inputs.append(output_sentence_input)


    for i in range(len(output_sentences)):
        input_sentences.append(input[i])


    return input_sentences, output_sentences, output_sentences_inputs


input_sentences, output_sentences, output_sentences_inputs = tagger(join_eng, join_ceb, NUM_SENTENCES)

print("num samples input:", len(input_sentences))
print("num samples output:", len(output_sentences))
print("num samples output input:", len(output_sentences_inputs))

print(input_sentences[5])
print(output_sentences[5])
print(output_sentences_inputs[5])

input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(input_sentences)
input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)

word2idx_inputs = input_tokenizer.word_index
print('Total unique words in the input: %s' % len(word2idx_inputs))

max_input_len = max(len(sen) for sen in input_integer_seq)
print("Length of longest sentence in input: %g" % max_input_len)

# ===============================================

output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(output_sentences + output_sentences_inputs)
output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)

word2idx_outputs = output_tokenizer.word_index
print('Total unique words in the output: %s' % len(word2idx_outputs))

num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)
print("Length of longest sentence in the output: %g" % max_out_len)

# ==================================================
encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)
print("encoder_input_sequences.shape:", encoder_input_sequences.shape)
print("encoder_input_sequences[172]:", encoder_input_sequences[172])
# ==================================================
decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
print("decoder_input_sequences.shape:", decoder_input_sequences.shape)
print("decoder_input_sequences[172]:", decoder_input_sequences[172])

# ==================================================
# WORD EMBEDDINGS
from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open(r'/Users/airadomingo/Documents/Capstone/glove/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

# =====================================================

num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
for word, index in word2idx_inputs.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# ======================================================

embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=max_input_len)


# ======================================================
# CREATE MODEL

decoder_targets_one_hot = np.zeros((
        len(input_sentences),
        max_out_len,
        num_words_output
    ),
    dtype='float32'
)

print(decoder_targets_one_hot.shape)

decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')

for i, d in enumerate(decoder_output_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1


encoder_inputs_placeholder = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LSTM_NODES, return_state=True)

encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]


decoder_inputs_placeholder = Input(shape=(max_out_len,))

decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)


# COMPILE
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs_placeholder,
  decoder_inputs_placeholder], decoder_outputs)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

from keras.utils import plot_model
plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

# ==================================================
history = model.fit(
    [encoder_input_sequences, decoder_input_sequences],
    decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
)

model.save_weights('saved_weights.hdf5', overwrite=True)

# SUMMARIZE HISTORY FOR ACCURACY
plt.figure()
plt.plot(history.history['accuracy'], label = 'train')
plt.plot(history.history['val_accuracy'], label = 'test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.title('Accuracy vs Epoch')
plt.show()

# SUMMARIZE HISTORY FOR LOSS
plt.figure()
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.title('Loss vs Epoch')
plt.show()


# MODIFYING MODEL FOR PREDICTIONS
encoder_model = Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = Input(shape=(LSTM_NODES,))
decoder_state_input_c = Input(shape=(LSTM_NODES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)


decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

plot_model(decoder_model, to_file='model_plot_dec.png', show_shapes=True, show_layer_names=True)


# ====================================================================



# =====================================================================
# MAKING PREDICTIONS

idx2word_input = {v:k for k, v in word2idx_inputs.items()}
idx2word_target = {v:k for k, v in word2idx_outputs.items()}

def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)



# TESTING MODEL

i = np.random.choice(len(input_sentences))
input_seq = encoder_input_sequences[i:i+1]
translation = translate_sentence(input_seq)
print('-')
print('Input:', input_sentences[i])
print('Response:', translation)


# input_sentence = word_tokenize(input_seq)
# input_sentence = [input_sentence]
# translation = word_tokenize(translation)
weights=(1.0, 0, 0, 0)
chencherry = SmoothingFunction()
score = sentence_bleu(input_sentences[i], translation, smoothing_function=chencherry.method1)
print('BLEU Score:', score)



