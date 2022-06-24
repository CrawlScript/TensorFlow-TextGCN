# coding=utf-8
import os

from tf_geometric.utils import tf_utils
from tf_geometric.utils.graph_utils import convert_edge_to_directed
from tf_sparse import SparseMatrix

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from collections import Counter

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
import tensorflow as tf
import tf_geometric as tfg
import pickle

data_dir = "datasets/rt-polarity"

lr = 1e-4
l2_coef = 3e-3

texts = []
labels = []
for fname, label in [["rt-polarity.pos", 1], ["rt-polarity.neg", 0]]:
    fpath = os.path.join(data_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(line.strip())
            labels.append(label)

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)


class PMIModel(object):

    def __init__(self):
        self.word_counter = None
        self.pair_counter = None

    def get_pair_id(self, word0, word1):
        pair_id = tuple(sorted([word0, word1]))
        return pair_id

    def fit(self, sequences, window_size):

        self.word_counter = Counter()
        self.pair_counter = Counter()
        num_windows = 0
        for sequence in tqdm(sequences):
            for offset in range(len(sequence) - window_size):
                window = sequence[offset:offset + window_size]
                num_windows += 1
                for i, word0 in enumerate(window):
                    self.word_counter[word0] += 1
                    for j, word1 in enumerate(window[i + 1:]):
                        pair_id = self.get_pair_id(word0, word1)
                        self.pair_counter[pair_id] += 1

        for word, count in self.word_counter.items():
            self.word_counter[word] = count / num_windows
        for pair_id, count in self.pair_counter.items():
            self.pair_counter[pair_id] = count / num_windows

    def transform(self, word0, word1):
        prob_a = self.word_counter[word0]
        prob_b = self.word_counter[word1]
        pair_id = self.get_pair_id(word0, word1)
        prob_pair = self.pair_counter[pair_id]

        if prob_a == 0 or prob_b == 0 or prob_pair == 0:
            return 0

        pmi = np.log(prob_pair / (prob_a * prob_b))
        # print(word0, word1, pmi)
        pmi = np.maximum(pmi, 0.0)
        # print(pmi)
        return pmi


def build_word_graph(num_words, embedding_size, pmi_model):
    x = tf.Variable(tf.random.truncated_normal([num_words, embedding_size], stddev=1 / np.sqrt(embedding_size)),
                    dtype=tf.float32, name="word_embedding")
    edges = []
    edge_weight = []
    for (word0, word1) in pmi_model.pair_counter.keys():
        pmi = pmi_model.transform(word0, word1)
        if pmi > 0:
            edges.append([word0, word1])
            edge_weight.append(pmi)
            edges.append([word1, word0])
            edge_weight.append(pmi)
    edge_index = np.array(edges).T
    return tfg.Graph(x=x, edge_index=edge_index, edge_weight=edge_weight)


def build_combined_graph(word_graph, sequences, embedding_size):
    num_words = word_graph.num_nodes

    edges = []
    edge_weight = []
    for i, sequence in enumerate(sequences):
        doc_node_index = num_words + i
        for word in sequence:
            edges.append([doc_node_index, word])  # only directed edge
            edge_weight.append(1.0)  # use BOW instaead of TF-IDF

    edge_index = np.array(edges).T

    edge_index = np.concatenate([word_graph.edge_index, edge_index], axis=1)
    edge_weight = np.concatenate([word_graph.edge_weight, edge_weight], axis=0)

    # edge_index, [edge_weight] = convert_edge_to_directed(edge_index, edge_props=[edge_weight], merge_modes=["sum"])

    x = tf.zeros([len(sequences), embedding_size], dtype=tf.float32)

    def x_func():
        return tf.concat([word_graph.x, x], axis=0)

    num_nodes = num_words + len(sequences)
    adj = SparseMatrix(edge_index, value=edge_weight, shape=[num_nodes, num_nodes])

    return x_func, adj


# building PMI model is time consuming, using cache to optimize
pmi_cache_path = "cached_pmi_model.p"
if os.path.exists(pmi_cache_path):
    with open(pmi_cache_path, "rb") as f:
        pmi_model = pickle.load(f)
else:
    pmi_model = PMIModel()
    pmi_model.fit(train_sequences, window_size=6)
    with open(pmi_cache_path, "wb") as f:
        pickle.dump(pmi_model, f)

embedding_size = 150
num_words = len(tokenizer.word_index) + 1
word_graph = build_word_graph(num_words, embedding_size, pmi_model)
train_x_func, train_adj = build_combined_graph(word_graph, train_sequences, embedding_size)
test_x_func, test_adj = build_combined_graph(word_graph, test_sequences, embedding_size)

print(word_graph)

num_classes = 2


class GCNModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn0 = tfg.layers.GCN(num_classes)
        self.dropout = keras.layers.Dropout(0.5)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, edge_weight = inputs
        h = self.dropout(x, training=training)
        h = self.gcn0([h, edge_index, edge_weight], cache=cache)
        return h


model = GCNModel()
train_cache = model.gcn0.build_cache_by_adj(train_adj)
test_cache = model.gcn0.build_cache_by_adj(test_adj)



def forward(x_func, adj, cache=None, training=False):
    x = x_func()
    logits = model([x, adj.index, adj.value], cache=cache, training=training)
    logits = logits[num_words:]
    return logits


@tf_utils.function
def compute_loss(logits, labels):
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=tf.one_hot(labels, depth=num_classes)
    )
    mean_loss = tf.reduce_mean(losses)
    return mean_loss


@tf_utils.function
def train_step():
    with tf.GradientTape() as tape:
        logits = forward(train_x_func, train_adj, cache=train_cache, training=True)
        mean_loss = compute_loss(logits, train_labels)

        l2_vars = [var for var in tape.watched_variables() if "kernel" in var.name or "embedding" in var.name]
        l2_losses = [tf.nn.l2_loss(var) for var in l2_vars]
        l2_loss = tf.add_n(l2_losses)
        loss = mean_loss + l2_coef * l2_loss

    vars = tape.watched_variables()
    grads = tape.gradient(mean_loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return loss



optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
for step in tqdm(range(1000)):

    loss = train_step()

    if step % 10 == 0:
        logits = forward(test_x_func, test_adj, cache=test_cache)
        preds = tf.argmax(logits, axis=-1)
        corrects = tf.cast(tf.equal(preds, test_labels), tf.float32)
        accuracy = tf.reduce_mean(corrects)
        print("step = {}\tloss = {}\ttest_accuracy = {}".format(step, loss, accuracy))
