import numpy as np
import random
import json
import os
import time

from faker import Faker
import babel
from babel.dates import format_date

import tensorflow as tf

import tensorflow.contrib.legacy_seq2seq as seq2seq
from sklearn.model_selection import train_test_split

fake = Faker()
fake.seed(42)
random.seed(42)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY', ]
# change this if you want it to work with only a single language
LOCALES = babel.localedata.locale_identifiers()
LOCALES = [lang for lang in LOCALES if 'en' in str(lang)]


def create_date():
    """
        Creates some fake dates 
        :returns: tuple containing 
                  1. human formatted string
                  2. machine formatted string
                  3. date object.
    """
    dt = fake.date_object()

    # wrapping this in a try catch because
    # the locale 'vo' and format 'full' will fail
    try:
        human = format_date(dt,
                            format=random.choice(FORMATS),
                            locale=random.choice(LOCALES))

        case_change = random.randint(0, 3)  # 1/2 chance of case change
        if case_change == 1:
            human = human.upper()
        elif case_change == 2:
            human = human.lower()

        machine = dt.isoformat()
    except AttributeError as e:
        return None, None, None

    return human, machine  # , dt


data = [create_date() for _ in range(50000)]

print(data[:5])

x = [x for x, y in data]
y = [y for x, y in data]

# Creates a set of all unique chars in x along with the space character
u_characters = set(' '.join(x))
# Creates a dictionary in which the unique chars are zipped to a number in the range of length of x, So each char has a number attached to it
char2numX = dict(zip(u_characters, range(len(u_characters))))

u_characters = set(' '.join(y))
char2numY = dict(zip(u_characters, range(len(u_characters))))

char2numX['<PAD>'] = len(char2numX)
num2charX = dict(zip(char2numX.values(), char2numX.keys()))
max_len = max([len(date) for date in x])

x = [[char2numX['<PAD>']] * (max_len - len(date)) + [char2numX[x_] for x_ in date] for date in x]
print(''.join([num2charX[x_] for x_ in x[4]]))
x = np.array(x)

char2numY['<GO>'] = len(char2numY)
num2charY = dict(zip(char2numY.values(), char2numY.keys()))

y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y]
print(''.join([num2charY[y_] for y_ in y[4]]))
y = np.array(y)

x_seq_length = len(x[0])
y_seq_length = len(y[0]) - 1


def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start + batch_size], y[start:start + batch_size]
        start += batch_size
# Main code involving tensorflow


batch_size = 128
nodes = 32
embed_size = 10
bidirectional = True

tf.reset_default_graph()
sess = tf.InteractiveSession()

# To feed the data
inputs = tf.placeholder(tf.int32, (None, x_seq_length), 'inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')

#Embedding layers
input_embedding = tf.Variable(tf.random_uniform((len(char2numX),embed_size),-1.0,1.0),'enc_embedding')
output_embedding  =tf.Variable(tf.random_uniform((len(char2numY),embed_size),-1.0,1.0),'dec_embedding')

#Lookups for the embedding layers, these will be actually used in the LSTM cell
date_input_embed    =   tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed   =   tf.nn.embedding_lookup(output_embedding,outputs)

# Neural network architecture for the encoder
with tf.variable_scope("encoding") as encoding_scope:

    if not bidirectional:
        # Simple forward LSTM units
        lstm_enc = tf.contrib.rnn.LSTMCell(nodes)
        # Using the lstm_enc, to attach to itself creating a dynamic rnn
        _ , last_state = tf.nn.dynamic_rnn(lstm_enc,inputs=date_input_embed,dtype=tf.float32)

    else:
        # Bidirectional LSTMs are used
        enc_fw_cell = tf.contrib.rnn.LSTMCell(nodes)
        enc_bw_cell = tf.contrib.rnn.LSTMCell(nodes)
        # Using the cells above to attach to themselves creating a dynamic rnn
        ((enc_fw_out,enc_bw_out), (enc_fw_final,enc_bw_final)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_fw_cell,
                                                                                                 cell_bw=enc_bw_cell,
                                                                                                 inputs=date_input_embed,
                                                                                                 dtype=tf.float32)
        enc_fin_c = tf.concat((enc_fw_final.c,enc_bw_final.c),1)
        enc_fin_h = tf.concat((enc_fw_final.h,enc_bw_final.c),1)
        last_state = tf.contrib.rnn.LSTMStateTuple(c=enc_fin_c,h=enc_fin_h)

# NN architecture for the decoder
with tf.variable_scope("decoding") as decoding_scope:

    if not bidirectional:
        # When only using the forward LSTM encoder
        lstm_dec = tf.contrib.rnn.LSTMCell(nodes)
    else:
        # When using bidirectional LSTM encoder
        lstm_dec = tf.contrib.rnn.LSTMCell(nodes*2)

    # Using the lstm_dec, to attach to itself creating a dynamic rnn
    dec_outputs,_ = tf.nn.dynamic_rnn(lstm_dec,inputs=date_output_embed,initial_state=last_state)

# Pass through a fully connected layer
logits = tf.layers.dense(dec_outputs, units=len(char2numY), use_bias=True)
print(len(char2numY))

# Connect outputs to
with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits,targets,tf.ones([batch_size,y_seq_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

# Train-Test split is done the generated dataset
X_train, X_test, Y_train, Y_test    =   train_test_split(x,y,test_size=0.33,random_state=42)

# Interactive session has begun, which marks the start of training by all the variables being initialized
sess.run(tf.global_variables_initializer())
# Total number of rounds
epochs = 10
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train,Y_train,batch_size)):
        _,batch_loss,batch_logits = sess.run([optimizer, loss, logits],
                                             feed_dict={inputs: source_batch,
                                                         outputs: target_batch[:,:-1],
                                                         targets: target_batch[:,1:]})
    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i,
                                                                                          batch_loss, accuracy,time.time() - start_time))

# Testing
source_batch , target_batch = next(batch_data(X_test,Y_test,batch_size))
dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
for i in range(y_seq_length):
    batch_logits = sess.run(logits,
                            feed_dict={inputs: source_batch,
                                       outputs: dec_input})
    prediction = batch_logits[:, -1].argmax(axis=-1)
    dec_input = np.hstack([dec_input, prediction[:, None]])

print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))