"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
"""
import numpy as np
from random import uniform
import sys

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def dtanh(x):
    return 1 - x*x
    
# data I/O
data = open('data/input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
std = 0.05

option = sys.argv[1]

# hyperparameters
emb_size = 32 
hidden_size = 256 # size of hidden layer of neurons
seq_length = 64 # number of steps to unroll the RNN for
learning_rate = 5e-2
max_updates = 500000

concat_size = emb_size + hidden_size

# model parameters
# char embedding parameters
Wex = np.random.randn(emb_size, vocab_size)*std # embedding layer

# LSTM parameters
Wf = np.random.randn(hidden_size, concat_size) * std # forget gate
Wi = np.random.randn(hidden_size, concat_size) * std # input gate
Wo = np.random.randn(hidden_size, concat_size) * std # output gate
Wc = np.random.randn(hidden_size, concat_size) * std # c term

bf = np.zeros((hidden_size, 1)) # forget bias
bi = np.zeros((hidden_size, 1)) # input bias
bo = np.zeros((hidden_size, 1)) # output bias
bc = np.zeros((hidden_size, 1)) # memory bias

# Output layer parameters
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
by = np.zeros((vocab_size, 1)) # output bias


def lossFun(inputs, targets, memory):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    hprev, cprev = memory
    xs, wes, hs, ys, ps, cs, zs, ins, c_s, = {}, {}, {}, {}, {}, {}, {}, {}, {}
    os, fs = {}, {}
    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)

    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1

        # convert word indices to word embeddings
        wes[t] = np.dot(Wex, xs[t])

        # LSTM cell operation
        # first concatenate the input and h
        zs[t] = np.row_stack((hs[t-1], wes[t]))

        # compute the forget gate
        f = sigmoid(np.dot(Wf, zs[t]) + bf)
        fs[t] = f
        # compute the input gate
        i = sigmoid(np.dot(Wi, zs[t]) + bi)
        ins[t] = i
        # compute the candidate memory
        c_ = np.tanh(np.dot(Wc, zs[t]) + bc)
        c_s[t] = c_

        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        cs[t] = f * cs[t-1] + i * c_

        # output gate
        o = sigmoid(np.dot(Wo, zs[t]) + bo)
        hs[t] = o * np.tanh(cs[t])
        os[t] = o

        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars
        ys[t] = np.dot(Why, hs[t]) + by 

        # softmax for probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) 

        # cross-entropy loss
        loss += -np.log(ps[t][targets[t],0]) 

    # backward pass: compute gradients going backwards
    # Here we allocate memory for the gradients
    dWex, dWhy = np.zeros_like(Wex), np.zeros_like(Why)
    dby = np.zeros_like(by)
    dWf, dWi, dWc, dWo = np.zeros_like(Wf), np.zeros_like(Wi),np.zeros_like(Wc), np.zeros_like(Wo)
    dbf, dbi, dbc, dbo = np.zeros_like(bf), np.zeros_like(bi),np.zeros_like(bc), np.zeros_like(bo)

    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])

    # back propagation through time starts here
    for t in reversed(range(len(inputs))):

        o = os[t]
        z = zs[t]
        c = cs[t]
        h = hs[t]
        i = ins[t]
        c_ = c_s[t]
        f = fs[t]

        # back-prop from the output layer 
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into y.
        dWhy += np.dot(dy, h.T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext # backprop into h

        # LSTM BACKWARD FROM HERE

        # backward through the output gate
        do = dh * np.tanh(c)
        # backward through sigmoid in o
        do = dsigmoid(o) * do 
        dWo += np.dot(do, z.T)
        dbo += do

        # next we backprop through the memory cell c

        # this is the gradient from the cell of the previous layer

        # because hs[t] = o * np.tanh(cs[t]) 
        # so the gradient w.r.t c should be
        dc = dh * o * dtanh(np.tanh(c)) + dcnext

        # gradient of the candidate
        dc_ = dc * i
        dc_ = dc_ * dtanh(c_s[t])
        dWc += np.dot(dc_, z.T)
        dbc += dc_

        # gradient w.r.t to the input gate
        di = dc * c_s[t]
        di = dsigmoid(i) * di
        dWi += np.dot(di, z.T)
        dbi += di

        # finally gradient w.r.t to the forget gate
        df = dc * cs[t-1]
        df = dsigmoid(f) * df
        dWf += np.dot(df, z.T)
        dbf += df 

        # now we can backprop to the concatenated input between input and h
        dz = np.dot(Wf.T, df ) + np.dot(Wi.T, di) + np.dot(Wc.T, dc_) + np.dot(Wo.T, do)

        # because of concatenation
        dhnext = dz[:hidden_size, :]

        # gradient w.r.t the previous cell (which will carry the gradient further to the past)
        dcnext = f * dc

        de = dz[hidden_size:hidden_size + emb_size:, :]

        # embedding backprop 
        dWex += np.dot(de, xs[t].T)
  
    # clip to mitigate exploding gradients
    for dparam in [dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby]:
        np.clip(dparam, -5, 5, out=dparam) 

    gradients = (dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby)
    memory = (hs[len(inputs)-1], cs[len(inputs)-1])

    return loss, gradients, memory


def sample(memory, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  h, c = memory
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    we = np.dot(Wex, x)
    # h = np.tanh(np.dot(Wxh, we) + np.dot(Whh, h) + bh)
    z = np.row_stack((h, we))

    f = sigmoid(np.dot(Wf, z) + bf)
    # compute the input gate
    i = sigmoid(np.dot(Wi, z) + bi)
    # compute the candidate memory
    c_ = np.tanh(np.dot(Wc, z) + bc)

    # new memory: applying forget gate on the previous memory
    # and then adding the input gate on the candidate memory
    c = f * c + i * c_

    # output gate
    o = sigmoid(np.dot(Wo, z) + bo)
    h = o * np.tanh(c)

    y = np.dot(Why, h) + by

    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())

    index = ix
    x = np.zeros((vocab_size, 1))
    x[index] = 1
    ixes.append(index)
  return ixes

if option == 'train':

    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(Wex), np.zeros_like(Why)
    mby = np.zeros_like(by) 

    mWf, mWi, mWo, mWc = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wo), np.zeros_like(Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bo), np.zeros_like(bc)

    smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
    
    while True:
      # prepare inputs (we're sweeping from left to right in steps seq_length long)
      if p+seq_length+1 >= len(data) or n == 0: 
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        cprev = np.zeros((hidden_size,1))
        p = 0 # go from start of data
      inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
      targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

      # sample from the model now and then
      if n % 100 == 0:
        sample_ix = sample((hprev, cprev), inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print ('----\n %s \n----' % (txt, ))

      # forward seq_length characters through the net and fetch gradient
      loss, gradients, memory = lossFun(inputs, targets, (hprev, cprev))
      
      hprev, cprev = memory
      dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if n % 100 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
      
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by], 
                                    [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby], 
                                    [mWf, mWi, mWo, mWc, mbf, mbi, mbo, mbc, mWex, mWhy, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += seq_length # move data pointer
      n += 1 # iteration counter 
      n_updates += 1
      if n_updates >= max_updates:
        break

elif option == 'gradcheck':

    p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    delta = 1e-5

    hprev = np.zeros((hidden_size,1))
    cprev = np.zeros((hidden_size,1))

    memory = (hprev, cprev)

    loss, gradients, hprev = lossFun(inputs, targets, memory)
    dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients

    for weight, grad, name in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by], 
                                   [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWo, dWhy, dby], 
                                   ['Wf', 'Wi', 'Wo', 'Wc', 'bf', 'bi', 'bo', 'bc', 'Wex', 'Why', 'by']):


      # assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
      str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
      assert (weight.shape == grad.shape), str_

      print(name)
      for i in range(weight.size):
      
        # evaluate cost at [x + delta] and [x - delta]
        w = weight.flat[i]
        weight.flat[i] = w + delta
        loss_positive, _, _ = lossFun(inputs, targets, memory)
        weight.flat[i] = w - delta
        loss_negative, _, _ = lossFun(inputs, targets, memory)
        weight.flat[i] = w # reset old value for this parameter
        # fetch both numerical and analytic gradient
        grad_analytic = grad.flat[i]
        grad_numerical = (loss_positive - loss_negative) / ( 2 * delta )
        rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic + 1e-9)
        print ('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
        # rel_error should be on order of 1e-7 or less