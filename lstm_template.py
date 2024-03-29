"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
"""
import numpy as np
from random import uniform
import sys


# Since numpy doesn't have a function for sigmoid
# We implement it manually here
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The derivative of the sigmoid function
def dsigmoid(y):
    return y * (1 - y)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# The derivative of the tanh function
def dtanh(x):
    return 1 - x*x


# The numerically stable softmax implementation
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# data I/O
data = open('data/input.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))  # unique characters in the data.
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))

# a python dictionary to map each character to an index from 0 - (vocab_size-1).
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
std = 0.1

option = 'train'  # sys.argv[1], 'train' or 'gradcheck'

# hyperparameters
emb_size = 4
hidden_size = 32  # size of hidden layer of neurons
seq_length = 64  # number of steps to unroll the RNN for
learning_rate = 5e-2
max_updates = 500000

concat_size = emb_size + hidden_size

# model parameters
# char embedding parameters
Wex = np.random.randn(emb_size, vocab_size)*std  # embedding layer

# LSTM parameters
Wf = np.random.randn(hidden_size, concat_size) * std  # forget gate
Wi = np.random.randn(hidden_size, concat_size) * std  # input gate
Wo = np.random.randn(hidden_size, concat_size) * std  # output gate
Wc = np.random.randn(hidden_size, concat_size) * std  # c term

bf = np.zeros((hidden_size, 1))  # forget bias
bi = np.zeros((hidden_size, 1))  # input bias
bo = np.zeros((hidden_size, 1))  # output bias
bc = np.zeros((hidden_size, 1))  # memory bias

# Output layer parameters
Why = np.random.randn(vocab_size, hidden_size)*0.01  # hidden to output
by = np.zeros((vocab_size, 1))  # output bias


def forward(i_inputs, i_targets, i_memory):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """

    # The LSTM is different than the simple RNN that it has two memory cells
    # so here you need two different hidden layers
    i_hprev, i_cprev = i_memory

    # Here you should allocate some variables to store the activations during forward
    # One of them here is to store the hiddens and the cells
    hs, cs, xs, wes, zs, os, ps, ys = {}, {}, {}, {}, {}, {}, {}, {}

    hs[-1] = np.copy(i_hprev)
    cs[-1] = np.copy(i_cprev)

    i_loss = 0
    # forward pass
    for t in range(len(i_inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][i_inputs[t]] = 1

        # convert word indices to word embeddings
        wes[t] = np.dot(Wex, xs[t])
        # print("wes: ", wes[t].shape)

        # LSTM cell operation
        # first concatenate the input and h
        # This step is irregular (to save the amount of matrix multiplication we have to do)
        # I will refer to this vector as [h X]
        zs[t] = np.row_stack((hs[t-1], wes[t]))

        # YOUR IMPLEMENTATION should begin from here

        # compute the forget gate
        f_gate = sigmoid(np.dot(Wf, zs[t]) + bf)

        # compute the input gate
        # i_gate = sigmoid (W_i \cdot [h X] + b_i)
        i_gate = sigmoid(np.dot(Wi, zs[t]) + bi)

        # compute the candidate memory
        # \hat{c} = tanh (W_c \cdot [h X] + b_c])
        c_hat = tanh(np.dot(Wc, zs[t]) + bc)

        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        # c_new = f_gate * prev_c + i_gate * \hat{c}
        cs[t] = f_gate * cs[t-1] + i_gate * c_hat

        # output gate
        # o_gate = sigmoid (Wo \cdot [h X] + b_o)
        o_gate = sigmoid(np.dot(Wo, zs[t]) + bo)

        # new hidden state for the LSTM
        # h = o_gate * tanh(c_new)
        hs[t] = o_gate * tanh(cs[t])

        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars

        # o = Why \cdot h + by
        os[t] = np.dot(Why, hs[t]) + by

        # softmax for probabilities for next chars
        # p = softmax(o)
        ps[t] = softmax(os[t])

        # cross-entropy loss
        # cross entropy loss at time t:
        # create an one hot vector for the label y

        ys[t] = np.zeros((vocab_size, 1))
        ys[t][i_targets[t]] = 1

        loss_t = np.sum(-np.log(ps[t]) * ys[t])

        i_loss += loss_t

        # and then cross-entropy (see the elman-rnn file for the hint)

    # define your activations
    i_memory = (hs[len(i_inputs)-1], cs[len(i_inputs)-1])
    i_activations = (xs, cs, hs, os, ps, ys, wes)

    return i_loss, i_activations, i_memory


def backward(i_activations, clipping=True):

    # backward pass: compute gradients going backwards
    # Here we allocate memory for the gradients
    i_dWex, i_dWhy = np.zeros_like(Wex), np.zeros_like(Why)
    i_dby = np.zeros_like(by)
    i_dWf, i_dWi, i_dWc, i_dWo = np.zeros_like(Wf), np.zeros_like(Wi),np.zeros_like(Wc), np.zeros_like(Wo)
    i_dbf, i_dbi, i_dbc, i_dbo = np.zeros_like(bf), np.zeros_like(bi),np.zeros_like(bc), np.zeros_like(bo)

    xs, cs, hs, os, ps, ys, wes = i_activations

    zs = {}
    # similar to the hidden states in the vanilla RNN
    # We need to initialize the gradients for these variables

    dc_future = np.zeros_like(cs[0])
    f_gate_future = np.zeros_like(bf)
    do_gate_pre_future = np.zeros_like(bo)
    dc_hat_pre_future = np.zeros_like(bc)
    di_gate_pre_future = np.zeros_like(bi)
    df_gate_pre_future = np.zeros_like(bf)

    # back propagation through time starts here
    for t in reversed(range(len(inputs))):

        zs[t] = np.row_stack((hs[t - 1], wes[t]))

        hs_row, hs_col = hs[t-1].shape

        do = ps[t] - ys[t]

        i_dWhy += np.dot(do, hs[t].T)
        i_dby += do
        # print("Wo.T: ", Wo.T.shape)
        # print("bo: ", bo.shape)
        # print("bc: ", bc.shape)
        # print("bi: ", bi.shape)
        # print("bf: ", bf.shape)
        dhnext = np.dot(Why.T, do) + np.dot(Wo.T, do_gate_pre_future)[0:hs_row, ...] \
                                   + np.dot(Wc.T, dc_hat_pre_future)[0:hs_row, ...] \
                                   + np.dot(Wi.T, di_gate_pre_future)[0:hs_row, ...] \
                                   + np.dot(Wf.T, df_gate_pre_future)[0:hs_row, ...]

        o_gate = sigmoid(np.dot(Wo, zs[t]) + bo)
        do_gate = dhnext * tanh(cs[t])
        dtanh_pro = o_gate * dhnext

        # output gate path
        dsigmoid_o_gate = dsigmoid(o_gate)
        do_gate_pre = dsigmoid_o_gate * do_gate
        i_dbo += do_gate_pre
        i_dWo += np.dot(do_gate_pre, zs[t].T)
        dzs = np.dot(Wo.T, do_gate_pre)

        # print("dwes: ", dwes.shape)
        # print("dxs: ", xs[t].shape)
        # memory (c) path
        f_gate = sigmoid(np.dot(Wf, zs[t]) + bf)
        c_hat = tanh(np.dot(Wc, zs[t]) + bc)
        i_gate = sigmoid(np.dot(Wi, zs[t]) + bi)

        dtanh_c = dtanh(tanh(cs[t]))
        dcnext = dc_future * f_gate_future + dtanh_c * dtanh_pro

        df_gate = dcnext * cs[t-1]
        di_gate = dcnext * c_hat
        dc_hat = i_gate * dcnext

        # forget gate path
        dsigmoid_f_gate = dsigmoid(f_gate)
        df_gate_pre = dsigmoid_f_gate * df_gate
        i_dbf += df_gate_pre
        i_dWf += np.dot(df_gate_pre, zs[t].T)
        dzs += np.dot(Wf.T, df_gate_pre)

        # input gate path
        dsigmoid_i_gate = dsigmoid(i_gate)
        di_gate_pre = dsigmoid_i_gate * di_gate
        i_dbi += di_gate_pre
        i_dWi += np.dot(di_gate_pre, zs[t].T)
        dzs += np.dot(Wi.T, di_gate_pre)

        # candidate memory (c hat) path
        dtanh_c_hat = dtanh(c_hat)
        dc_hat_pre = dtanh_c_hat * dc_hat
        i_dbc += dc_hat_pre
        i_dWc += np.dot(dc_hat_pre, zs[t].T)
        dzs += np.dot(Wc.T, dc_hat_pre)

        # delta xs[t]
        dwes = dzs[hs_row:, ...]
        i_dWex += np.dot(dwes, xs[t].T)

        dc_future = dcnext
        f_gate_future = f_gate
        do_gate_pre_future = do_gate_pre
        dc_hat_pre_future = dc_hat_pre
        di_gate_pre_future = di_gate_pre
        df_gate_pre_future = df_gate_pre

        # IMPLEMENT YOUR BACKPROP HERE
        # refer to the file elman_rnn.py for more details

    if clipping:
        # clip to mitigate exploding gradients
        for dparam in [i_dWex, i_dWf, i_dWi, i_dWo, i_dWc, i_dbf, i_dbi, i_dbo, i_dbc, i_dWhy, i_dby]:
            np.clip(dparam, -5, 5, out=dparam)

    i_gradients = (i_dWex, i_dWf, i_dWi, i_dWo, i_dWc, i_dbf, i_dbi, i_dbo, i_dbc, i_dWhy, i_dby)

    return i_gradients


def sample(i_memory, seed_ix, i_n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    (h, c) = i_memory
    # print(c.shape)
    x = np.zeros((vocab_size, 1))
    i_sample_ix = []
    x[seed_ix] = 1

    for t in range(i_n):
        wes = np.dot(Wex, x)
        z = np.row_stack((h, wes))

        f_gate = sigmoid(np.dot(Wf, z) + bf)

        i_gate = sigmoid(np.dot(Wi, z) + bi)

        c_hat = np.tanh(np.dot(Wc, z) + bc)

        c = f_gate * c + i_gate * c_hat

        o_gate = sigmoid(np.dot(Wo, z) + bo)

        h = o_gate * np.tanh(c)

        o = np.dot(Why, h) + by

        i_p = softmax(o)

        # the the distribution, we randomly generate samples:
        ix = np.random.multinomial(1, i_p.ravel())
        x = np.zeros((vocab_size, 1))

        for j in range(len(ix)):
            if ix[j] == 1:
                index = j
        x[index] = 1
        i_sample_ix.append(index)

    return i_sample_ix


if option == 'train':

    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(Wex), np.zeros_like(Why)
    mby = np.zeros_like(by) 

    mWf, mWi, mWo, mWc = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wo), np.zeros_like(Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bo), np.zeros_like(bc)

    smooth_loss = -np.log(1.0/vocab_size)*seq_length  # loss at iteration 0
    
    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p+seq_length+1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size, 1))  # reset RNN memory
            cprev = np.zeros((hidden_size, 1))
        p = 0  # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

        # sample from the model now and then
        if n % 100 == 0:
            sample_ix = sample((hprev, cprev), inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('----\n %s \n----' % (txt, ))

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = forward(inputs, targets, (hprev, cprev))
        gradients = backward(activations)

        hprev, cprev = memory
        dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0:
            print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                      [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                      [mWf, mWi, mWo, mWc, mbf, mbi, mbo, mbc, mWex, mWhy, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

        p += seq_length  # move data pointer
        n += 1  # iteration counter
        n_updates += 1
        if n_updates >= max_updates:
            break

elif option == 'gradcheck':

    p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    delta = 0.001

    hprev = np.zeros((hidden_size, 1))
    cprev = np.zeros((hidden_size, 1))

    memory = (hprev, cprev)

    loss, activations, _ = forward(inputs, targets, memory)
    gradients = backward(activations, clipping=False)
    dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients

    for weight, grad, name in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by], 
                                  [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                  ['Wf', 'Wi', 'Wo', 'Wc', 'bf', 'bi', 'bo', 'bc', 'Wex', 'Why', 'by']):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert(weight.shape == grad.shape), str_

        print(name)
        for i in range(weight.size):
      
            # evaluate cost at [x + delta] and [x - delta]
            w = weight.flat[i]
            weight.flat[i] = w + delta
            loss_positive, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w - delta
            loss_negative, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w  # reset old value for this parameter

            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / (2 * delta)

            # compare the relative error between analytical and numerical gradients
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)

            if rel_error > 0.01:
                print('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
