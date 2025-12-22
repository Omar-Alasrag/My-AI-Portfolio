# %%


# data link
# https://www.google.com/search?q=http://www.manythings.org/anki/spa-eng.zip

# emb link
# https://nlp.stanford.edu/projects/glove/

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    RepeatVector,
    Input,
    Embedding,
    concatenate,
    Lambda,
    Softmax,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.backend as K


MAX_N_WORDS = 20000
EMB_DIM = 100
# LOAD DATA
LATENT_DIM = 10
MAX_SEQUENCE_LEN = 50  # max input length
MAX_TARGET_LEN = 10  # max target length
NUM_SAMPLES = 20000

inputs = []
targets = []
targets_input = []

print("reading the data")
with open("spa.txt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= NUM_SAMPLES:
            break
        X, y = line.lower().strip().split("\t")[:2]
        inputs.append(X)
        targets_input.append("<sos> " + y)
        targets.append(y + " <eos>")

input_tokenizer = Tokenizer(MAX_N_WORDS, filters="")
output_tokenizer = Tokenizer(MAX_N_WORDS, filters="")
input_tokenizer.fit_on_texts(inputs)
output_tokenizer.fit_on_texts(targets_input + targets)

t_inpus = input_tokenizer.texts_to_sequences(inputs)
t_targets_input = input_tokenizer.texts_to_sequences(targets_input)
t_targets = input_tokenizer.texts_to_sequences(targets)

n_words = min(len(input_tokenizer.word_index), MAX_N_WORDS) + 1
word2idx = input_tokenizer.word_index

print("loading embeddings")
emb_mat = np.zeros((n_words, EMB_DIM))
with open("glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.rstrip().lower().split(" ")
        word = values[0]
        if word2idx.get(word) is not None and word2idx[word] < n_words:
            emb_mat[word2idx[word]] = np.array(values[1:], dtype=np.float32)

pad_inputs = pad_sequences(t_inpus, maxlen=MAX_SEQUENCE_LEN)
pad_targets_input = pad_sequences(t_targets_input, maxlen=MAX_TARGET_LEN)
pad_targets = pad_sequences(t_targets, maxlen=MAX_TARGET_LEN)

# encoder
encoder_input = Input((MAX_SEQUENCE_LEN,))
encoder_emb_layer = Embedding(n_words, EMB_DIM, weights=[emb_mat])
encoder_lstm = Bidirectional(LSTM(LATENT_DIM, "tanh", return_sequences=True))
encoder_emb_out = encoder_emb_layer(encoder_input)  # batch * max_seq * emb_dim
eo = encoder_lstm(encoder_emb_out)  # batch * max_seq * 2latent

# decoder
decoder_target_input = Input((MAX_TARGET_LEN,))
last_c = Input(shape=(LATENT_DIM,))
last_s = Input(shape=(LATENT_DIM,))

last_c_copy = last_c
last_s_copy = last_s
decoder_emb_layer = Embedding(n_words, EMB_DIM, weights=[emb_mat])
decoder_lstm = LSTM(LATENT_DIM, "tanh", return_state=True)
repeat_layer = RepeatVector(MAX_SEQUENCE_LEN)
attention_dense = Dense(10, "tanh")
attention_dense_2 = Dense(1, Softmax(axis=-2))
decoder_dense = Dense(n_words, "softmax")

decoder_emb_out = decoder_emb_layer(
    decoder_target_input
)  # batch * max_target_len * emb_dim
output = []

for t in range(MAX_TARGET_LEN):
    concat_input = concatenate([repeat_layer(last_s_copy), eo])
    x = attention_dense(concat_input)
    alphas = attention_dense_2(x)
    context = Lambda(lambda z: K.sum(z[0] * z[1], axis=1, keepdims=True))([eo, alphas])

    selected_target_input = Lambda(lambda x: x[:, t : t + 1])(decoder_emb_out)
    do, dh, dc = decoder_lstm(
        concatenate([selected_target_input, context]),
        initial_state=[last_s_copy, last_c_copy],
    )
    out = decoder_dense(do)
    output.append(out)
    last_s_copy = dh
    last_c_copy = dc


def stack_and_transpose(x):
    x = K.stack(x)
    x = K.permute_dimensions(x, pattern=(1, 0, 2))
    return x


output = stack_and_transpose(output)

model = Model([encoder_input, decoder_target_input, last_s, last_c], output)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

z = np.zeros((len(pad_inputs), LATENT_DIM))
print("training starts")
model.fit(
    [pad_inputs, pad_targets_input, z, z],
    pad_targets,
    batch_size=64,
    epochs=1,
    verbose=1,
)
print("training ending")

# encoder model
encoder_model = Model(encoder_input, eo)

# decoder model
decoder_input_t = Input(shape=(1,))
encoder_out = Input(shape=(MAX_SEQUENCE_LEN, 2 * LATENT_DIM))
decoder_s = Input(shape=(LATENT_DIM,))
decoder_c = Input(shape=(LATENT_DIM,))
dec_emb_t = decoder_emb_layer(decoder_input_t)

concat_input_inf = concatenate([RepeatVector(MAX_SEQUENCE_LEN)(decoder_s), encoder_out])
x_inf = attention_dense(concat_input_inf)
alphas_inf = attention_dense_2(x_inf)
context_inf = Lambda(lambda z: K.sum(z[0] * z[1], axis=1, keepdims=True))(
    [encoder_out, alphas_inf]
)
decoder_lstm_input_inf = concatenate([dec_emb_t, context_inf])
do_inf, s_out_inf, c_out_inf = decoder_lstm(
    decoder_lstm_input_inf, initial_state=[decoder_s, decoder_c]
)
decoder_out_inf = decoder_dense(do_inf)

decoder_model = Model(
    [decoder_input_t, encoder_out, decoder_s, decoder_c],
    [decoder_out_inf, s_out_inf, c_out_inf],
)

idx2word = {v: k for k, v in output_tokenizer.word_index.items()}


def translate(sentence, max_len=MAX_TARGET_LEN):
    seq = input_tokenizer.texts_to_sequences([sentence.lower()])
    seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LEN)
    enc_out = encoder_model.predict(seq)

    sos = output_tokenizer.word_index["<sos>"]
    eos = output_tokenizer.word_index["<eos>"]

    word = sos
    s = np.zeros((1, LATENT_DIM))
    c = np.zeros((1, LATENT_DIM))

    result = []
    for _ in range(max_len):
        word_input = np.array([[word]])
        probs, s, c = decoder_model.predict([word_input, enc_out, s, c], verbose=0)
        word = np.argmax(probs[0, 0])
        if word == eos:
            break
        result.append(idx2word.get(word, ""))
    return " ".join(result)


print(translate("how are you"))
print(translate("i am hungry"))
