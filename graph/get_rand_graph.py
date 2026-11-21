import random

def get_rand_graph(n_inp, n_con, n_hid, n_out, funcs, use_all_input = False):
	'''
	Just run the script in the commad line to see an example of how it works!
	Here we generate a random "graph" given as two lists and an integer.
	The graph is executed by the code in "graph.py" file.

	Arguments:
		n_inp - number of input    nodes.
		n_con - number of constant nodes.
		n_hid - number of hidden   nodes.
		n_out - number of output   nodes.
		funcs - a list of possible funtion as tuples ('func_name', n_args).
		use_all_input - if True, each input node (x1, x2, ...) is used by at least one node.
			If False (default), it is not guaranteed (some of the input nodes can stay unused).

	Output:
		Triplet: "names", "graph", "n_consts" ("names" and "graph" are both lists of the same length).
		names - is a list of the following form: ['h1', 'h2', 'h3', 'o1', 'o2']
		graph - a list of nodes. Each node has the following structure: ['add', 'h2', 'x1'] or ['mul', 'c1', 'x5']
			We can say that that the elements of the "names" list specify the output node
			and the "graph" list specify the operation and input nodes.
		n_consts - number of constants used by function.

	Algorithm:
		Go from "right to left" over all hidden nodes and make sure that output of each node is used somewhere
		(either by one of the next hidden nodes or by one of the outputs).
		If "use_all_input" is True, each input node (x1, x2, ...) is put into a randomly chosen vacant argument.
			This needs at least "n_inp" vacant arguments after the previous step. If the randomly chosen
			functions do not have enough arguments, the functions are chosen again.
			If it is not possible even with the functions that have the most arguments, ValueError is raised.
		Go from "left to right" over hidden and output nodes and make sure that all arguments a filled.

		We make sure that no node use only constants as inputs:
			If the last argument needs to be specified and all the other arguments are constants then
			the last argument will be either x or one of the hidden nodes.
		This guaranties that all the hidden nodes and all the output nodes (directly on indirectly) depend
			on an non empty set of input nodes.

		We cannot guaranty that all constants will be used. First, beacause they are chosen randomly.
		For example it can happen that, just by chance, only the first constant is chosen.
		Second, something there are not enough vacant arguents to use all the constant.
		In the end we rename the constans such that they are called from 1 to N (c1, c2, c3, ...).
	'''

	n_nodes = n_hid + n_out

	# Each hidden node and (if requested) each input node needs its own argument.
	if use_all_input:
		max_args = max([func[1] for func in funcs])
		if n_nodes * max_args < n_hid + n_inp:
			raise ValueError('not enough arguments in the graph to use all the hidden and input nodes')

	# for each output and hidden node choose a function
	while True:
		graph = []
		# empty_args is a list of tuples (node_ind, arg_ind) indexing arguments
		# the first (most left) hidden node gets index = 0
		# the last output node (most right) gets index = n_hid + n_out - 1
		empty_args = []
		for i in range(n_nodes):
			func = random.choice(funcs)
			n_args = func[1]
			graph.append([func[0]] + ['' for k in range(n_args)])
			for j in range(n_args):
				empty_args.append((i, j))
		# if all the input nodes have to be used, we choose the functions again
		# until they have enough arguments.
		if not use_all_input or len(empty_args) >= n_hid + n_inp:
			break

	# specify names of the nodes
	names  = ['h' + str(i + 1) for i in range(n_hid)]
	names += ['o' + str(i + 1) for i in range(n_out)]

	# go from right to left over the hidden nodes and and put the current node as someone's input
	for i in range(n_hid):
		j = n_hid - i - 1 # [n_hid - 1, n_hid - 2, ..., 0]
		arg = random.choice([x for x in empty_args if x[0] > j])
		empty_args.remove(arg)
		# put the current hidden node as input for the chose argument
		graph[arg[0]][1 + arg[1]] = names[j]

	# create list of inputs, constants and hidden nodes
	xs = ['x' + str(i + 1) for i in range(n_inp)]
	cs = ['c' + str(i + 1) for i in range(n_con)]
	hs = ['h' + str(i + 1) for i in range(n_hid)]

	# put each input node into a randomly chosen vacant argument (an input node can be used by any node)
	if use_all_input:
		for x in xs:
			arg = random.choice(empty_args)
			empty_args.remove(arg)
			graph[arg[0]][1 + arg[1]] = x

	# loop over the arguments of a given node of a given graph
	# calculate the number of non specified arguments,
	# arguments that use input node, hidden node, constant node.
	# Return True if there is exactly one non-specifed argument left
	# and no input and hidden nodes are used.
	# Only constants are used or this is a node with one argument.
	def only_consts(graph, i):
		e_n, x_n, h_n, c_n = 0, 0, 0, 0
		for x in graph[i][1:]:
			if x == '':
				e_n += 1
			elif x.startswith('x'):
				x_n += 1
			elif x.startswith('h'):
				h_n += 1
			elif x.startswith('c'):
				c_n += 1
		if e_n == 1 and x_n + h_n == 0:
			return True
		else:
			return False

	used_cs = []
	# Loop over arguments that are not specified (empty)
	for arg in empty_args:
		i, j = arg
		case = only_consts(graph, i)

		# the last (or onyl) argument and no dependency so far
		# make sure that we have one argument which is not constant
		if case:
			options = xs + hs[:i]
		else:
			options = xs + cs + hs[:i]

		# randomly choose an option
		val = random.choice(options)
		graph[i][j + 1] = val

		# collect used constants
		if val.startswith('c'):
			used_cs.append(val)

	used_cs = list(set(used_cs))
	used_cs.sort()
	old2new = dict([(used_cs[i], 'c' + str(i+1)) for i in range(len(used_cs))])

	for i in range(len(graph)):
		node = graph[i]
		for j in range(len(node) - 1):
			val = graph[i][j + 1]
			graph[i][j + 1] = old2new.get(val, val)


	n_consts = len(used_cs)
	return names, graph, n_consts

if __name__ == '__main__':

	n_inp = 5
	n_con = 4
	n_hid = 7
	n_out = 2
	funcs = [('add',2), ('mul',2), ('exp', 1), ('log', 1)]
	names, graph, n_const = get_rand_graph(n_inp, n_con, n_hid, n_out, funcs)
	print('names =', names)
	print('graph = ', graph)
	print('n_const = ', n_const)

	#from print_graph import print_graph
	#print_graph(names, graph)
