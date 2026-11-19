import random

def get_rand_graph(n_inp, n_con, n_hid, n_out, funcs):
	'''
	Just run the script in the commad line to see an example of how it works!
	Here we generate a random "graph" given as two lists and an integer.
	The graph is executed by the code in "Graph.py" file.

	Arguments:
		n_inp - number of input    nodes.
		n_con - number of constant nodes.
		n_hid - number of hidden   nodes.
		n_out - number of output   nodes.
		funcs - a list of possible funtion as tuples ('func_name', n_args).

	Output:
		Triplet: "names", "graph", "n_consts" ("names" and "graph" are both lists of the same length).
		names - is a list of the following form: ['h1', 'h2', 'h3', 'o1', 'o2']
		graph - a list of nodes. Each node has the following structure: ['plus', 'h2', 'x1'] or ['mul', 'c1', 'x5']
			We can say that that the elements of the "names" list specify the output node
			and the "graph" list specify the operation and input nodes.
		n_consts - number of constants used by function.

	Algorithm:
		Go from "right to left" over all hidden nodes and make sure that output of each node is used somewhere
		(either by one of the next hidden nodes or by one of the outputs).
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

	# for each output and hidden node choose a function
	graph = []
	n_nodes = n_hid + n_out
	# empty_args is a list of tuples (node_ind, arg_ind) indexing arguments
	empty_args = []
	for i in range(n_nodes):
		func = random.choice(funcs)
		n_args = func[1]
		graph.append([func[0]] + ['' for k in range(n_args)])
		for j in range(n_args):
			empty_args.append((i, j))

	# specify names of the nodes
	names  = ['h' + str(i + 1) for i in range(n_hid)]
	names += ['o' + str(i + 1) for i in range(n_out)]

	# go from right to left over the hidden nodes and and put the current node as someone's input
	for i in range(n_hid):
		j = n_hid - i - 1
		arg = random.choice([x for x in empty_args if x[0] > j])
		empty_args.remove(arg)
		graph[arg[0]][1 + arg[1]] = names[j]

	# create list of inputs, constants and hidden nodes
	xs = ['x' + str(i + 1) for i in range(n_inp)]
	cs = ['c' + str(i + 1) for i in range(n_con)]
	hs = ['h' + str(i + 1) for i in range(n_hid)]

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
	for arg in empty_args:
		i, j = arg
		case = only_consts(graph, i)
		if case:
			options = xs + hs[:i]
		else:
			options = xs + cs + hs[:i]
		val = random.choice(options)
		graph[i][j + 1] = val
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
	funcs = [('plus',2), ('mul',2), ('exp', 1), ('log', 1)]
	names, graph, n_const = get_rand_graph(n_inp, n_con, n_hid, n_out, funcs)
	print('names =', names)
	print('graph = ', graph)
	print('n_const = ', n_const)

	#from print_graph import print_graph
	#print_graph(names, graph)
