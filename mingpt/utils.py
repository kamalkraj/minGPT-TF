import random

import numpy as np
import tensorflow as tf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def top_k_logits(logits, k):
    v, ix = tf.math.top_k(logits, k)
    out = tf.identity(logits).numpy()
    out[out < v.numpy()[:, [-1]]] = -float('Inf')
    return out

def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.block_size
    for k in range(steps):
        x_cond = x if x.shape[1] <= block_size else x[:, -block_size:] # crop context if needed
        logits = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = tf.nn.softmax(logits, axis=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = tf.random.categorical(logits,1)
        else:
            _, ix = tf.math.top_k(probs, k=1)
        # append to the sequence and continue
        x = tf.concat((x,ix), axis=1)

    return x
