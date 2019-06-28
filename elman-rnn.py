"""
Implementation of the character-level Elman RNN model.
Written by Ngoc-Quan Pham based on Andreij Karparthy's lecture Cs231n.
BSD License
"""
import numpy as np
from random import uniform

# data I/O

# should be simple plain text file. I provided the sample from "Hamlet - Shakespeares"
data = open('data/input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyper-parameters deciding the network size
emb_size = 5 # word/character embedding size
seq_length = 32 # number of steps to unroll the RNN for the truncated back-propagation algorithm
hidden_size = 10
# learning rate for the Adagrad algorithm. (this one is not 'optimized', only required to make the model learn)
learning_rate = 1e-1

# model parameters
# Here we initialize the parameters based an random uniform distribution, with the std of 0.01

# word embedding: each character in the vocabulary is mapped to a vector with $emb_size$ neurons
# Transform one-hot vectors to embedding X
Wex = np.random.randn(emb_size, vocab_size) * 0.01

# weight to transform input X to hidden H
Wxh = np.random.randn(hidden_size, emb_size) * 0.01

# weight to transform previous hidden states H_{t-1} to hidden H_t
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden

# Output layer: transforming the hidden states H to output layer
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output

# The biases are typically initialized as zeros. But sometimes people init them with uniform distribution too.
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

# These variables are momentums for the Adagrad algorithm
# Each parameter in the network needs one momentum correspondingly
mWex, mWxh, mWhh, mWhy = np.zeros_like(Wex), np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)


def forward(inputs, labels, memory):

	prev_h = memory
	"""
	# dictionaries to store the activations over time
	# note from back-propagation implementation:
	# back-propagation uses dynamic programming to estimate gradients efficiently
	# so we need to store the activations over the course of the forward pass
	# in the backward pass we will use the activations to compute the gradients
	# (otherwise we will need to recompute them)
	"""

	# those variables stand for:
	# xs: inputs to the RNNs at timesteps (embeddings)
	# cs: characters at timesteps
	# hs: hidden states at timesteps
	# ys: output layers at timesteps
	# ps: probability distributions at timesteps
	xs, cs, hs, os, ps, ys = {}, {}, {}, {}, {}, {}

	# the first memory (before training) is the previous (or initial) hidden state
	hs[-1] = np.copy(prev_h)

	# the loss will be accumulated over time
	loss = 0

	for t in range(len(inputs)):

		# one-hot vector representation for character input at time t
		cs[t] = np.zeros((vocab_size,1))
		cs[t][inputs[t]] = 1

		# transform the one hot vector to embedding
		# x = Wemb x c
		xs[t] = np.dot(Wex, cs[t])

		# computation for the hidden state of the network
		# H = tanh ( Wh . H + Wx . x )
		h_pre_activation = np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh
		hs[t] = np.tanh(h_pre_activation)

		# output layer:
		# this is the unnormalized log probabilities for next chars (across all chars in the vocabulary)
		os[t] = np.dot(Why, hs[t]) + by

		# softmax layer to get normalized probabilities:
		exp_ = np.exp(os[t])
		ps[t] = exp_ / np.sum(exp_)

		# cross entropy loss at time t:
		ys[t] = np.zeros((vocab_size, 1))
		ys[t][labels[t]] = 1

		loss_t = np.sum(-np.log(ps[t]) * ys[t])
		# loss_t = -np.log(ps[t][labels[t],0])

		loss += loss_t

	# packaging the activations to use in the backward pass
	activations = (xs, cs, hs, os, ps, ys)
	last_hidden = hs[-1]
	return loss, activations, last_hidden


def backward(activations):
	"""
	during the backward pass we follow the track of the forward pass
	the activations are needed so that we can avoid unnecessary re-computation
	"""

	# Gradient initialization
	# Each parameter has a corresponding gradient (of the loss with respect to that gradient)
	dWex, dWxh, dWhh, dWhy = np.zeros_like(Wex), np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	dbh, dby = np.zeros_like(bh), np.zeros_like(by)

	xs, cs, hs, os, ps, ys = activations

	# here we need the gradient w.r.t to the hidden layer at the final time step
	# since this hidden layer is not connected to any future (final time step)
	# then we can initialize it as zero vectors
	dh = np.zeros_like(hs[0])

	# the backward pass starts from the final step of the chain in the forward pass
	for t in reversed(range(len(inputs))):

		# first, we need to compute the gradients of the variable closest to the loss function, which is the softmax output p
		# but here I skip it directly to the gradients of the unnormalized scores o
		# basically dL / do = p - y
		# from the cross entropy gradients. (the explanation is a bit too long to write here)
		do = ps[t] - ys[t]

		# the gradients w.r.t to the weights and the bias that were used to create o[t]
		dWhy += np.dot(do, hs[t].T)
		dby += do

		# because h is connected to both o and the next h, we sum the gradients up
		dh = np.dot(Why.T, do) + dh

		# backprop through the activation function (tanh)
		dtanh_h = 1 - hs[t] * hs[t]
		dh_pre_activation = dtanh_h * dh # because h = tanh(h_pre_activation)

		# next, since  H = tanh ( Wh . H + Wx . x + bh )
		# we use dh to backprop to dWh and dWx

		# gradient of the bias and weight, this is similar to dby and dWhy
		# for the H term
		dbh += dh_pre_activation
		dWhh += np.dot(dh_pre_activation, hs[t-1].T)
		# we need this term for the recurrent connection (previous bptt step needs this)
		dh = np.dot(Whh.T, dh_pre_activation)

		# similarly for the x term
		dWxh += np.dot(dh_pre_activation, xs[t].T)

		# backward through the embedding
		dx = np.dot(Wxh.T, dh_pre_activation)

		# finally backward to the embedding projection
		dWex += np.dot(dx, cs[t].T)

	for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
		np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

	gradients = (dWex, dWxh, dWhh, dWhy, dbh, dby)
	return gradients


def sample(h, seed_ix, n):
	""" 
	sample a sequence of integers from the model 
	h is memory state, seed_ix is seed letter for first time step
	"""
	x = np.zeros((vocab_size, 1))
	x[seed_ix] = 1
	ixes = []
	for t in range(n):
		we = np.dot(Wex, x)
		h = np.tanh(np.dot(Wxh, we) + np.dot(Whh, h) + bh)
		y = np.dot(Why, h) + by
		p = np.exp(y) / np.sum(np.exp(y))
		ix = np.random.choice(range(vocab_size), p=p.ravel())
		
		index = ix
		x = np.zeros((vocab_size, 1))
		x[index] = 1
		ixes.append(index)
	return ixes


option = 'gradcheck' # train or grad_check

if option == 'train':

	n, p = 0, 0
	smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
	while True:
		# prepare inputs (we're sweeping from left to right in steps seq_length long)
		if p+seq_length+1 >= len(data) or n == 0:
			hprev = np.zeros((hidden_size, 1))  # reset RNN memory
			p = 0 # go from start of data
		inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
		targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

		# sample from the model now and then
		if n % 1000 == 0:
			sample_ix = sample(hprev, inputs[0], 200)
			txt = ''.join(ix_to_char[ix] for ix in sample_ix)
			print ('----\n %s \n----' % (txt, ))

		# forward seq_length characters through the net and fetch gradient
		# loss, gradients, hprev = lossFun(inputs, targets, hprev)
		loss, activations, memory = forward(inputs, targets, hprev)
		gradients = backward(activations)
		dWex, dWxh, dWhh, dWhy, dbh, dby = gradients
		smooth_loss = smooth_loss * 0.999 + loss * 0.001
		if n % 1000 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress

		# perform parameter update with Adagrad
		for param, dparam, mem in zip([Wex, Wxh, Whh, Why, bh, by],
									  [dWex, dWxh, dWhh, dWhy, dbh, dby],
									  [mWex, mWxh, mWhh, mWhy, mbh, mby]):
			mem += dparam * dparam
			param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

		p += seq_length # move data pointer
		n += 1 # iteration counter

elif option == 'gradcheck':

	p = 0
	inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
	targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

	delta = 1e-5

	hprev = np.zeros((hidden_size,1))
	memory = hprev

	loss, activations, memory = forward(inputs, targets, hprev)
	gradients = backward(activations)
	dWex, dWxh, dWhh, dWhy, dbh, dby = gradients

	for weight, grad, name in zip([Wex, Wxh, Whh, Why, bh, by], [dWex, dWxh, dWhh, dWhy, dbh, dby], ['Wex', 'Wxh', 'Whh', 'Why', 'bh', 'by']):

		str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
		assert (weight.shape == grad.shape), str_

		print(name)
		for i in range(weight.size):

			# evaluate cost at [x + delta] and [x - delta]
			w = weight.flat[i]
			weight.flat[i] = w + delta
			loss_positive, _, _ = forward(inputs, targets, hprev)
			weight.flat[i] = w - delta
			loss_negative, _, _ = forward(inputs, targets, hprev)
			weight.flat[i] = w # reset old value for this parameter
			# fetch both numerical and analytic gradient
			grad_analytic = grad.flat[i]
			grad_numerical = (loss_positive - loss_negative) / ( 2 * delta )
			rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic + 1e-9)
			print ('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
			# rel_error should be on order of 1e-7 or less

