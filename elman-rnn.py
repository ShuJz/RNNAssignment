"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
from random import uniform

# data I/O
data = open('data/input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
emb_size = 32 
hidden_size = 100 # size of hidden layer of neurons
seq_length = 32 # number of steps to unroll the RNN for
learning_rate = 1e-1
option = 'gradcheck' # train or grad_check

# model parameters
Wex = np.random.randn(emb_size, vocab_size)*0.01 # word emedding 
Wxh = np.random.randn(hidden_size, emb_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias
		
def lossFun(inputs, targets, hprev):
	"""
	inputs,targets are both list of integers.
	hprev is Hx1 array of initial hidden state
	returns the loss, gradients on model parameters, and last hidden state
	"""
	xs, wes, hs, ys, ps= {}, {}, {}, {}, {}
	hs[-1] = np.copy(hprev)
	loss = 0
	# forward pass
	for t in range(len(inputs)):
		xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
		xs[t][inputs[t]] = 1
		wes[t] = np.dot(Wex, xs[t])
		hs[t] = np.tanh(np.dot(Wxh, wes[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
		ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
		loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
	# backward pass: compute gradients going backwards
	dWex, dWxh, dWhh, dWhy = np.zeros_like(Wex), np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	dbh, dby = np.zeros_like(bh), np.zeros_like(by)
	dhnext = np.zeros_like(hs[0])
	for t in reversed(range(len(inputs))):
		dy = np.copy(ps[t])
		dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
		dWhy += np.dot(dy, hs[t].T)
		dby += dy
		dh = np.dot(Why.T, dy) + dhnext # backprop into h
		dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
		dbh += dhraw
		dWxh += np.dot(dhraw, wes[t].T)
		dWhh += np.dot(dhraw, hs[t-1].T)
		dhnext = np.dot(Whh.T, dhraw)
		de = np.dot(Wxh.T, dhraw)
		dWex += np.dot(de, xs[t].T)
	for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
		np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
	return loss, dWex, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

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

if option == 'train':

		n, p = 0, 0
		mWex, mWxh, mWhh, mWhy = np.zeros_like(Wex), np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
		mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
		smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
		while True:
			# prepare inputs (we're sweeping from left to right in steps seq_length long)
			if p+seq_length+1 >= len(data) or n == 0: 
				hprev = np.zeros((hidden_size,1)) # reset RNN memory
				p = 0 # go from start of data
			inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
			targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

			# sample from the model now and then
			if n % 1000 == 0:
				sample_ix = sample(hprev, inputs[0], 200)
				txt = ''.join(ix_to_char[ix] for ix in sample_ix)
				print ('----\n %s \n----' % (txt, ))

			# forward seq_length characters through the net and fetch gradient
			loss, dWex, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
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

		loss, dWex, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)

		for weight, grad, name in zip([Wex, Wxh, Whh, Why, bh, by], [dWex, dWxh, dWhh, dWhy, dbh, dby], ['Wex', 'Wxh', 'Whh', 'Why', 'bh', 'by']):
			
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