'''
	Just run the script in the command line to see an example of how it works!

	Class that represents a "smooth" version of the graphs from "graph.py".

	In "graph.py" each hidden and each output node is one fixed function,
	and each argument of this function is one fixed previous node.
	Here both choices are replaced by numbers (parameters):

		1. Each node is a weighted average of all the available functions:
			node = w1*f1(...) + w2*f2(...) + w3*f3(...) + ...
		The weights of a node are positive and sum up to 1.0.
		They are calculated as a softmax of scores.

		2. Each argument of each function is a linear combination of all the previous nodes:
			arg = c + k1*n1 + k2*n2 + k3*n3 + ...
		Each argument of each function of each node has its own "k"s and its own "c".

	There are no nodes for constants (c1, c2, ...). Their role is played by the "c" above.

	Order of the nodes:
		We imagine the nodes written one above the other:

			output nodes     o1, o2, ...       (top)
			hidden nodes     ..., h3, h2, h1
			input nodes      ..., x3, x2, x1
			constant node    always 1.0        (bottom)

		A hidden node sees everything below it.
		An output node sees all hidden and all input nodes and the constant node (but not other output nodes).

	Parameters of the model:

		'z'  - Matrix of scores.
			One row per computable node (first hidden nodes, then output nodes, as in "names").
			One column per function (in the order given at the instantiation).

		'w'  - Matrix of weights, softmax of 'z' (each row sums up to 1.0).
			It is calculated from 'z' and cannot be set directly.

		'h1', 'h2', ..., 'o1', 'o2', ... - One matrix per computable node.
			One column per argument, for all the functions together.
			For example, for the functions [add, exp, smif] we have 2 + 1 + 3 = 6 columns:
				add: columns 0, 1;   exp: column 2;   smif: columns 3, 4, 5
			One row per node that can be seen by the given node. From top to bottom:
				previous hidden nodes (the nearest one first: ..., h2, h1)
				input nodes (..., x2, x1)
				the constant node (the last row, it contains the "c"s)
			For example the rows of 'h3' in a model with two inputs are: h2, h1, x2, x1, 1.0
			The method "get_row_names" returns these names.

	Usage (see __main__ part of the code):

		g = SmoothGraph(n_inp = 2, n_hid = 3, n_out = 1, funcs = ['add', 'mul', 'exp'])
		g['z']                      # read parameters
		g['h2'][0, 1] = 0.5         # change parameters
		node2val = g.get_node2val({'x1':x1, 'x2':x2})
		node2val['o1']

		* At instantiation all the parameters are set randomly (Gaussian distribution).
		* The method "set_graph" sets the parameters such that the model reproduces
		a usual graph (given as "names" and "graph", see "get_rand_graph.py").

	Invalid values:
		If a function returns nan or inf (for example log(0) or 1/0), this value is replaced by 0.0
		before the weighted average is calculated.
'''

import inspect
import numpy as np
from graph import Graph

class SmoothGraph:

	def __init__(self, n_inp, n_hid, n_out, funcs, sigma = 1.0, seed = None):
		'''
		Arguments:
			n_inp - number of input  nodes (x1, x2, ...).
			n_hid - number of hidden nodes (h1, h2, ...).
			n_out - number of output nodes (o1, o2, ...).
			funcs - list of the functions to use. Either names: ['add', 'mul', 'exp']
					or tuples as used by "get_rand_graph": [('add',2), ('mul',2), ('exp',1)]
					The functions are those defined in "graph.py".
			sigma - standard deviation of the random initial values of the parameters.
			seed  - seed for the random initial values (optional).
		'''
		self.n_inp = n_inp
		self.n_hid = n_hid
		self.n_out = n_out

		# functions and their numbers of arguments
		fname2func = Graph([], []).fname2func
		self.fnames = []
		self.funcs  = []
		self.n_args = []
		for func in funcs:
			fname = func if isinstance(func, str) else func[0]
			if fname not in fname2func:
				raise ValueError('unknown function: ' + str(fname))
			if fname in self.fnames:
				raise ValueError('function is given twice: ' + str(fname))
			n_args = len(inspect.signature(fname2func[fname].get_value).parameters)
			if not isinstance(func, str) and func[1] != n_args:
				raise ValueError('function ' + fname + ' has ' + str(n_args) + ' arguments, not ' + str(func[1]))
			self.fnames.append(fname)
			self.funcs.append(fname2func[fname])
			self.n_args.append(n_args)
		self.n_funcs = len(self.fnames)
		self.n_cols  = sum(self.n_args)
		# index of the first column of each function in the node matrices
		self.first_col = [sum(self.n_args[:i]) for i in range(self.n_funcs)]

		# names of the computable nodes and of the rows of their matrices
		self.names  = ['h' + str(i + 1) for i in range(n_hid)]
		self.names += ['o' + str(i + 1) for i in range(n_out)]
		xs = ['x' + str(n_inp - i) for i in range(n_inp)]   # [..., 'x2', 'x1']
		self.row_names = {}
		for i in range(n_hid):
			hs = ['h' + str(i - k) for k in range(i)]       # [..., 'h2', 'h1'] (only previous)
			self.row_names['h' + str(i + 1)] = hs + xs + ['1']
		for i in range(n_out):
			hs = ['h' + str(n_hid - k) for k in range(n_hid)]
			self.row_names['o' + str(i + 1)] = hs + xs + ['1']

		# random initial values of the parameters
		rng = np.random if seed is None else np.random.RandomState(seed)
		self.params = {}
		self.params['z'] = rng.normal(0.0, sigma, size = (len(self.names), self.n_funcs))
		for name in self.names:
			n_rows = len(self.row_names[name])
			self.params[name] = rng.normal(0.0, sigma, size = (n_rows, self.n_cols))

	#---------------------------------------------------------------------------

	def __getitem__(self, name):
		'''
		Gives the parameters by name: g['z'], g['w'], g['h1'], g['o1'], ...
		'''
		if name == 'w':
			z = self.params['z']
			e = np.exp(z - np.max(z, axis = 1, keepdims = True))
			return e / np.sum(e, axis = 1, keepdims = True)
		return self.params[name]

	def __setitem__(self, name, value):
		'''
		Sets the parameters by name: g['z'] = ..., g['h1'] = ...
		'''
		if name == 'w':
			raise KeyError("'w' is calculated from 'z', set 'z' instead")
		value = np.array(value, dtype = float)
		if value.shape != self.params[name].shape:
			raise ValueError("'" + name + "' should have the shape " + str(self.params[name].shape))
		self.params[name] = value

	def get_row_names(self, name):
		'''
		Names of the rows of the matrix of the given node.
		For example for 'h3' and two inputs: ['h2', 'h1', 'x2', 'x1', '1']
		'''
		return self.row_names[name][:]

	def get_col_names(self):
		'''
		Names of the columns of the node matrices (the same for all the nodes).
		For example for the functions [add, exp]: ['add:1', 'add:2', 'exp:1']
		'''
		return [self.fnames[i] + ':' + str(k + 1) for i in range(self.n_funcs) for k in range(self.n_args[i])]

	#---------------------------------------------------------------------------

	def get_node2val(self, node2val):
		'''
		Takes a dict mapping names of the input nodes to their values.
			For example: {'x1':1.2, 'x2':3.7}
			The values can be numbers or numpy arrays.
		Extends the input dictionary by adding mapping from hidden and output nodes to their values.
		'''
		w = self['w']

		# all the inputs should have the same shape
		xs = ['x' + str(i + 1) for i in range(self.n_inp)]
		vals = np.broadcast_arrays(*[np.asarray(node2val[x], dtype = float) for x in xs])
		node2arr = dict(zip(xs, vals))

		# invalid values (nan, inf) are expected here, they are replaced by 0.0
		with np.errstate(all = 'ignore'):
			for i in range(len(self.names)):
				name = self.names[i]
				m = self.params[name]
				# values of the visible nodes, shape: (n_rows - 1, shape of input)
				prev = np.stack([node2arr[r] for r in self.row_names[name][:-1]], axis = 0)
				# values of all the arguments, shape: (n_cols, shape of input)
				args = np.tensordot(m[:-1].T, prev, axes = 1)
				args = args + m[-1].reshape((self.n_cols,) + (1,) * (prev.ndim - 1))

				val = 0.0
				for f in range(self.n_funcs):
					c = self.first_col[f]
					fval = self.funcs[f].get_value(*args[c : c + self.n_args[f]])
					fval = np.where(np.isfinite(fval), fval, 0.0)
					val = val + w[i, f] * fval
				# the sum of very large values can also become invalid
				val = np.where(np.isfinite(val), val, 0.0)
				node2arr[name] = val
				node2val[name] = val

		return node2val

	#---------------------------------------------------------------------------

	def set_graph(self, names, graph, consts = None, score = 50.0):
		'''
		Sets the parameters such that the model reproduces the given usual graph.

		Arguments:
			names, graph - a graph in the format generated by the "get_rand_graph" function.
				It should have the same numbers of hidden and output nodes as the model
				and use only the functions of the model.
			consts - dict mapping the constants of the graph to their values, for example {'c1':0.5, 'c2':-2.0}
			score  - the score given to the function of each node (the other functions get 0.0).
				The weight of the function is then 1.0 up to about exp(-score).
				So, with a usual number the other functions still add a very small value (about 1e-22 for 50.0).
				With score = np.inf the function of the node gets the score 0.0 and the other functions
				get -inf. Then the weights are exactly 1.0 and 0.0 and the graph is reproduced exactly
				(but such scores cannot be changed in small steps anymore).

		For each node:
			'z' - the function of the node gets "score", all the other functions get 0.0 (see above for np.inf)
			matrix of the node - everything is 0.0, except:
				argument is an input or a hidden node -> 1.0 in the row of this node.
				argument is a constant -> the value of the constant in the last row.
		'''
		if consts is None:
			consts = {}
		if list(names) != self.names:
			raise ValueError('the graph should have the nodes ' + str(self.names) + ', not ' + str(list(names)))

		new_params = {'z' : np.zeros((len(self.names), self.n_funcs))}
		if np.isinf(score):
			new_params['z'][:] = -np.inf
			score = 0.0
		for i in range(len(self.names)):
			name = self.names[i]
			fname = graph[i][0]
			args  = graph[i][1:]
			if fname not in self.fnames:
				raise ValueError('function ' + str(fname) + ' (node ' + name + ') is not in the model')
			f = self.fnames.index(fname)
			if len(args) != self.n_args[f]:
				raise ValueError('function ' + fname + ' (node ' + name + ') should have ' + str(self.n_args[f]) + ' arguments')
			new_params['z'][i, f] = score

			rows = self.row_names[name]
			m = np.zeros((len(rows), self.n_cols))
			for k in range(len(args)):
				col = self.first_col[f] + k
				if args[k] in rows[:-1]:
					m[rows.index(args[k]), col] = 1.0
				elif args[k] in consts:
					m[-1, col] = consts[args[k]]
				elif args[k].startswith('c'):
					raise KeyError('no value for the constant ' + args[k])
				else:
					raise ValueError('node ' + name + ' cannot use ' + args[k] + ' as an argument')
			new_params[name] = m

		self.params = new_params


if __name__ == '__main__':

	from get_rand_graph import get_rand_graph

	#---------------------------------------------------------------------------
	''' A random model.
	'''
	g = SmoothGraph(n_inp = 2, n_hid = 2, n_out = 1, funcs = ['add', 'mul', 'exp'], seed = 0)
	np.set_printoptions(precision = 3, suppress = True)
	print('names of nodes:', g.names)
	print('z =')
	print(g['z'])
	print('w =')
	print(g['w'])
	print('rows of h2:', g.get_row_names('h2'))
	print('cols of h2:', g.get_col_names())
	print('h2 =')
	print(g['h2'])
	x1 = np.array([0.1, 0.2, 0.3])
	x2 = np.array([1.0, 2.0, 3.0])
	node2val = g.get_node2val({'x1':x1, 'x2':x2})
	print('o1 =', node2val['o1'])
	print('')

	#---------------------------------------------------------------------------
	''' Reproduce a usual graph.
	'''
	n_inp, n_con, n_hid, n_out = 2, 3, 3, 2
	funcs = [('add',2), ('subtract',2), ('mul',2), ('tanh',1), ('sin',1), ('smif',3)]
	names, graph, n_consts = get_rand_graph(n_inp, n_con, n_hid, n_out, funcs)
	consts = dict([('c' + str(i + 1), np.random.normal()) for i in range(n_consts)])
	print('graph =', graph)
	print('consts =', consts)

	inp = {'x1':x1, 'x2':x2}
	inp.update(consts)
	v1, grads = Graph(names, graph).get_node2val(inp)

	g = SmoothGraph(n_inp, n_hid, n_out, funcs)
	g.set_graph(names, graph, consts)
	v2 = g.get_node2val({'x1':x1, 'x2':x2})

	for name in names:
		print(name, 'usual graph:', v1[name] + 0.0*x1, 'smooth graph:', v2[name])
