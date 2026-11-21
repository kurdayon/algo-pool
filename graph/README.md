# graph

Small tools for working with math functions written as graphs (DAGs, directed acyclic graphs).

There are four scripts:

- `graph.py` has the `Graph` class. It takes a graph, computes its values for given inputs, and can run backpropagation to get gradients. It can also run the same graph step by step over a sequence, like a recurrent network.
- `get_rand_graph.py` has the `get_rand_graph` function. It builds a random graph in the format that `Graph` understands.
- `mutate_graph.py` has the `mutate_graph` function. It takes a graph and returns a slightly changed (mutated) graph that is still valid.
- `smooth_graph.py` has the `SmoothGraph` class. It is a "smooth" version of a graph, where the choice of function and the choice of arguments are numbers (parameters) instead of fixed picks.

The only dependency is `numpy` (needed by `graph.py` and `smooth_graph.py`).

## How a graph is defined

A graph is two lists of the same length:

```python
names = ['h1', 'h2', 'o1']
graph = [
    ['mul', 'x1', 'c1'],   # h1 = x1 * c1
    ['mul', 'x2', 'x2'],   # h2 = x2 * x2
    ['add', 'h1', 'h2'],   # o1 = h1 + h2
]
```

- `names[i]` is the name of the node that is computed.
- `graph[i]` says how to compute it: the function name first, then the names of its arguments.

Rules:

- List only the nodes that are computed. Inputs and constants (here `x1`, `x2`, `c1`) are not listed. You give their values later.
- Order matters. Nodes are computed from first to last, so a node can only use inputs, constants, and nodes that come earlier in the list.
- For plain evaluation and backpropagation you can pick any names. The sequence mode needs the fixed names `x1, x2, ...`, `c1, c2, ...` and `o1, o2, ...` (see below).
- The graph is not checked. An unknown node name or function name gives a `KeyError` when you evaluate.

### Available functions

| Name | Arguments | What it computes |
|---|---|---|
| `add` | 2 | `x1 + x2` |
| `subtract` | 2 | `x1 - x2` |
| `mul` | 2 | `x1 * x2` |
| `divide` | 2 | `x1 / x2` |
| `max` | 2 | larger of `x1`, `x2` |
| `min` | 2 | smaller of `x1`, `x2` |
| `power` | 2 | `sign(x1) * abs(x1) ** x2` |
| `exp` | 1 | `exp(x1)` |
| `log` | 1 | `sign(x1) * log(abs(x1))` |
| `sqrt` | 1 | `sign(x1) * sqrt(abs(x1))` |
| `square` | 1 | `x1 * x1` |
| `neg` | 1 | `-x1` |
| `abs` | 1 | `abs(x1)` |
| `sin` | 1 | `sin(x1)` |
| `cos` | 1 | `cos(x1)` |
| `tanh` | 1 | `tanh(x1)` |
| `sigmoid` | 1 | `1 / (1 + exp(-x1))` |
| `relu` | 1 | `x1` if it is positive, else `0` |
| `smif` | 3 | smooth "if", see below |

Notes:

- All functions have gradients, so all of them can be used with `calc_grads=True`, with `back`, and in the sequence mode.
- `power`, `log` and `sqrt` are changed versions that also accept negative numbers. They keep the sign of `x1`.
- `smif` is a smooth "if". The second argument decides which of the other two wins: `x1 * (1 - w) + x3 * w` with `w = (tanh(x2) + 1) / 2`. If `x2` is very negative the result is `x1`. If `x2` is very positive the result is `x3`. At `x2 = 0` it is the average of the two.
- Use exactly the names from the table. For example `add` and `mul` work, `plus` and `multiply` do not.

Corner cases of the gradients:

- `max` and `min`: the derivative is 1 for the argument that wins and 0 for the other one. If both arguments are equal, the first one gets the 1.
- `abs` and `relu`: at `x1 = 0` the derivative is taken as `1` for `abs` and `0` for `relu`.
- `divide` (at `x2 = 0`), `log`, `sqrt` and `power` (at `x1 = 0`) have no finite derivative at these points. You get `inf` or `nan` there, and numpy prints a warning.

To add your own function, write a small class with `get_value(...)` and `get_grads(...)` and register it in `fname2func` inside `Graph.__init__`. `get_grads` must return a list with one derivative per argument, also when there is only one argument.

## Using `Graph`

### Evaluate

```python
import numpy as np
from graph import Graph

g = Graph(names, graph)
values, grads = g.get_node2val({'x1': 5.0, 'x2': 6.0, 'c1': 7.0}, calc_grads=True)

values['o1']   # 71.0
```

- The input is a dictionary with a value for every input and constant.
- Values can be plain numbers or numpy arrays. With arrays, everything is computed element by element, so you can evaluate many data points in one call.
- `values` is the same dictionary you passed in, with all computed nodes added to it. Pass a copy if you want to keep the original.
- `grads` is empty unless `calc_grads=True`. Then it holds, for each node in `names`, the derivatives of that node with respect to its direct arguments. For the example above: `[[7.0, 5.0], [6.0, 6.0], [1.0, 1.0]]`. You normally do not read it yourself, you pass it to `back`.

### Backpropagation

```python
d = g.back({'o1': 1.0}, grads)

d['x1']   # 7.0   = d o1 / d x1
d['x2']   # 12.0  = d o1 / d x2
d['c1']   # 5.0   = d o1 / d c1
```

- The first argument is a dictionary with the starting gradient for the outputs. `{'o1': 1.0}` means "give me the derivatives of `o1`".
- The result is the same dictionary, filled with the derivative for every node, input and constant.
- Every output node must be in the starting dictionary. If the graph has `o1` and `o2` and you want derivatives of `o1` only, pass `{'o1': 1.0, 'o2': 0.0}`.
- Every hidden node must be used by some later node. Otherwise `back` fails with a `KeyError`.

### Gradients with respect to inputs or constants?

Both. `Graph` does not treat inputs (`x1, x2, ...`) and constants (`c1, c2, ...`) differently. For `get_node2val` and `back` they are all just names of values that you provide. After `back`, the result has a derivative for every name that is used in the graph:

- inputs (`x1, x2, ...`),
- constants (`c1, c2, ...`),
- hidden nodes (`h1, h2, ...`).

So in the example above `d['x1']` and `d['c1']` come out of the same call. This is what you need if you want to fit the constants: `d['c1']` tells how the output changes when `c1` changes.

Two details:

- A name that is not used anywhere in the graph gets no entry in the result (its derivative is zero).
- With arrays, the derivative has one value per element. If a constant is a single number and the inputs are arrays, `d['c1']` is an array with one value per data point. Sum it if you want the derivative of the sum over all data points.

### Sequence mode (recurrent)

The same graph can be applied step by step to a sequence. Part of the output of one step becomes part of the input of the next step.

```python
names = ['h1', 'h2', 'o1', 'o2']
graph = [
    ['add', 'x1', 'x2'],
    ['mul', 'x1', 'x2'],
    ['add', 'h1', 'x3'],
    ['add', 'h2', 'x4'],
]
g = Graph(names, graph)

Xs    = np.ones(shape=(4, 2))    # 4 time steps, 2 external inputs per step
cs    = np.array([])             # values of c1, c2, ... (none here)
h_ini = np.array([1.0, 1.0])     # starting state, size 2

node2val_list, grads_list = g.seq_eval(Xs, cs, h_ini)
```

How the names are wired, with a state of size `n` (`n = len(h_ini)`):

- `x1 ... xn` are the state. At the first step they get `h_ini`. At every later step they get `o1 ... on` from the step before.
- `x(n+1), x(n+2), ...` get the current row of `Xs`.
- `c1, c2, ...` get the values from `cs`.
- `o1 ... on` are the new state. Any further outputs (`o(n+1)`, ...) are just stored.

Do not confuse the state with the hidden nodes `h1, h2, ...` of the graph. The state lives in the `x` and `o` nodes.

`seq_eval` returns one dictionary of values per step and one gradient list per step. It always computes the gradients.

Backpropagation through the sequence:

```python
start = [{} for _ in range(4)]
start[-1] = {'o1': 1.0, 'o2': 0.0}     # derivatives of o1 at the last step

d_list = g.seq_back(start, grads_list, h_dim=2)

d_list[0]['x2']   # derivative of the last o1 with respect to x2 at the first step
```

- `start` holds the starting gradients of the outputs for each step. `h_dim` is the state size.
- At the last step you must give all outputs. At earlier steps the state outputs (`o1 ... on`) are filled in from the next step. Outputs that are not part of the state must be given by you at every step (use `0.0` if they do not matter).
- A starting gradient you put on a state output at an earlier step is overwritten, not added.

Also here you get derivatives for both inputs and constants, one dictionary per step:

- `d_list[t]['x3']` is the derivative with respect to the external input at step `t`.
- `d_list[0]['x1']` is the derivative with respect to the starting state (`h_ini`).
- `d_list[t]['c1']` is only the part of the derivative that comes from the use of `c1` at step `t`. The constants are the same at every step, so the full derivative with respect to `c1` is the sum over all steps: `sum(d['c1'] for d in d_list)`.

## Using `get_rand_graph`

```python
from get_rand_graph import get_rand_graph

funcs = [('add', 2), ('mul', 2), ('exp', 1)]    # (function name, number of arguments)
names, graph, n_consts = get_rand_graph(n_inp=2, n_con=3, n_hid=3, n_out=1, funcs=funcs)
```

Possible result:

```python
names    = ['h1', 'h2', 'h3', 'o1']
graph    = [['add', 'c1', 'x2'], ['exp', 'x1'], ['add', 'h1', 'h2'], ['mul', 'h3', 'x2']]
n_consts = 1
```

Arguments:

- `n_inp` - number of inputs (`x1, x2, ...`).
- `n_con` - the largest number of constants the graph may use.
- `n_hid` - number of hidden nodes (`h1, h2, ...`).
- `n_out` - number of outputs (`o1, o2, ...`).
- `funcs` - the functions to choose from. Use names that `Graph` knows (see the table).
- `use_all_input` (optional, default `False`) - if `True`, every input (`x1 ... x<n_inp>`) is used by at least one node.

What you can rely on:

- The result can be given directly to `Graph(names, graph)`.
- Every hidden node is used by a later hidden node or by an output.
- No node is computed from constants only, so every node depends on at least one input.
- `n_consts` tells how many constants the graph really uses. It can be smaller than `n_con`. The used constants are always named `c1 ... c<n_consts>`, so you know exactly which values to provide.
- With the default `use_all_input=False` it is not guaranteed that every input is used. With `use_all_input=True` it is.

Running `python get_rand_graph.py` prints one example graph.

### How a random graph is generated

Think of the nodes as written in one row from left to right: first the hidden nodes `h1, h2, ...`, then the outputs `o1, o2, ...`. Inputs and constants are on the left of everything. A node may only use what is on the left of it.

1. **Choose the functions.** Every hidden node and every output gets a random function from `funcs`. This fixes how many arguments each node has. All arguments are still empty.

2. **Go from right to left over the hidden nodes and make sure each one is used.** We start with the last hidden node and end with `h1`. For each hidden node we pick a random empty argument of some node on the right of it (a later hidden node or an output) and put the hidden node there. After this step the output of every hidden node is used as an input of some node on the right.

3. **Only with `use_all_input=True`: make sure each input is used.** Each input `x1, x2, ...` is put into a random argument that is still empty. An input can go to any node, because inputs are on the left of all nodes.

4. **Fill the remaining arguments.** We loop over the arguments that are still empty. For each one we choose at random among everything that is on the left of the node: an input, a constant, or an earlier hidden node. (For an output, all hidden nodes are on the left.)

5. **Rename the constants.** See below.

**No node uses only constants.** In step 4, when the last empty argument of a node is filled and all its other arguments are constants, the choice is made only among inputs and earlier hidden nodes. So every node gets at least one argument that is an input or a hidden node. Since every hidden node follows the same rule, every node depends, directly or through other nodes, on at least one input.

**The number of constants.** `n_con` is the largest number of constants that may be used. It is not guaranteed that all of them are used. First, the arguments are chosen at random, so some constants may simply never be picked. Second, there may be too few free arguments for all of them. This is why the function returns `n_consts`, the number of constants that are really used. At the end the used constants are renamed to `c1, c2, ...`, so the numbering starts from 1 and has no gaps. For example, if only `c1` and `c3` were picked, `c3` becomes `c2` and `n_consts` is 2.

**Enough arguments for `use_all_input=True`.** Every hidden node and every input needs its own argument (steps 2 and 3). If the functions chosen in step 1 have too few arguments in total, step 1 is repeated. If it is impossible even when every node gets the function with the most arguments, a `ValueError` is raised. For example, one node with one argument cannot use five inputs.

Other details:

- The same node can be used several times, also twice by the same node, for example `['mul', 'x2', 'x2']`.
- Outputs are never used as arguments of other nodes.
- "Depends on an input" is about the connections only. A node like `['subtract', 'h1', 'h1']` is connected to the inputs, but its value is always zero.

## Using `mutate_graph`

`mutate_graph` makes one random change in a graph and returns the new graph. It is meant for search methods that improve a graph step by step (for example evolution).

```python
from mutate_graph import mutate_graph

funcs = [('add', 2), ('mul', 2), ('exp', 1)]
names = ['h1', 'o1']
graph = [['mul', 'x1', 'c1'], ['exp', 'h1']]

names, graph, n_consts, info = mutate_graph(names, graph, n_inp=2, n_con=2, funcs=funcs)
```

Arguments:

- `names`, `graph` - the graph to mutate. These lists are not changed.
- `n_inp` - number of inputs (`x1, x2, ...`) the graph may use.
- `n_con` - the largest number of constants the graph may use.
- `funcs` - the functions to choose from, as in `get_rand_graph`.
- `kinds` (optional) - list of allowed mutation kinds. By default all four are allowed.
- `seed` (optional) - give a number to get the same mutation again.

The four kinds of mutation (one is picked at random):

| Kind | What happens | Example for `h1 = mul(x1, c1)`, `o1 = exp(h1)` |
|---|---|---|
| `change_op` | One node gets another function. If the new function needs fewer arguments, some old ones are dropped. If it needs more, new ones are added. | `h1 = add(x1, c1)` |
| `change_arg` | One argument of one node is replaced by another input, constant or earlier hidden node. | `h1 = mul(x1, x2)` |
| `add_node` | A new hidden node is put into an existing connection. If node B uses A, a new node N is made that uses A, and B now uses N. | `h1 = mul(x1, c1)`, `h2 = add(h1, x2)`, `o1 = exp(h2)` |
| `remove_node` | One hidden node is removed. Every node that used it gets one of its arguments instead. | `o1 = exp(x1)` |

What you can rely on after a mutation:

- The new graph is different from the old one.
- Every node uses only inputs, constants and earlier hidden nodes.
- Every hidden node is used by a later node.
- No node is computed from constants only.
- Hidden nodes are again named `h1, h2, ...` in order, and the used constants `c1 ... c<n_consts>` without gaps. So the result can go directly to `Graph(names, graph)` or into `mutate_graph` again.
- The outputs stay. Output nodes are never added or removed, but their function and arguments can change.

Because nodes are renamed, a node can have a different name after the mutation (for example the old `h3` becomes `h4` when a node is inserted before it). The same holds for constants, and this matters if you keep fitted values for them. `info` tells you what happened:

- `info['type']` - the kind of mutation.
- `info['node']` - the changed or added node (name in the new graph), or the removed node (name in the old graph).
- `info['const_map']` - old constant name to new constant name, for example `{'c1': 'c1', 'c3': 'c2'}`. A constant that is no longer used is not in the map.
- `info['new_const']` - the name of a constant that was not in the old graph, or `None`.

If no mutation of the allowed kinds is possible, a `ValueError` is raised. For example, `kinds=['remove_node']` fails on a graph without hidden nodes. With `kinds=['change_op', 'change_arg']` the number of nodes stays fixed.

Running `python mutate_graph.py` shows a few mutations in a row and then runs a stress test (30000 mutations, each result is checked).

## Using `SmoothGraph`

In a usual graph each node is one fixed function, and each argument is one fixed earlier node. In a smooth graph both choices are replaced by numbers:

- **Each node is a weighted average of all functions:** `node = w1*f1(...) + w2*f2(...) + w3*f3(...) + ...`. The weights of a node are positive and sum to 1. They are the softmax of scores.
- **Each argument is a linear combination of all visible nodes:** `arg = c + k1*n1 + k2*n2 + ...`. Every argument of every function of every node has its own `k`s and its own `c`.

There are no constant nodes (`c1, c2, ...`). The `c` in the linear combination takes their role.

```python
import numpy as np
from smooth_graph import SmoothGraph

g = SmoothGraph(n_inp=2, n_hid=3, n_out=1, funcs=['add', 'mul', 'exp'])

values = g.get_node2val({'x1': np.array([0.1, 0.2]), 'x2': np.array([1.0, 2.0])})
values['o1']
```

Arguments of `SmoothGraph`:

- `n_inp`, `n_hid`, `n_out` - number of inputs, hidden nodes and outputs.
- `funcs` - the functions to use, from the table above. Names (`['add', 'mul']`) or the pairs used by `get_rand_graph` (`[('add', 2), ('mul', 2)]`) both work. You do not have to use all functions.
- `sigma` (optional, default 1.0) - all parameters start as random numbers from a Gaussian distribution with this standard deviation.
- `seed` (optional) - give a number to get the same starting parameters again.

`get_node2val` works like in `Graph`: it takes a dictionary with the input values (numbers or numpy arrays) and adds the values of all hidden and output nodes to it. It returns only this dictionary. There are no gradients yet.

### Order of the nodes

Think of the nodes written one above the other:

```
output nodes     o1, o2, ...        (top)
hidden nodes     ..., h3, h2, h1
input nodes      ..., x3, x2, x1
constant node    always 1.0         (bottom)
```

A hidden node sees everything below it. An output node sees all hidden nodes, all inputs and the constant node, but not other outputs.

### Parameters

All parameters are read by name with `g[...]`:

| Name | Shape | What it is |
|---|---|---|
| `g['z']` | nodes x functions | Scores. One row per hidden and output node (hidden first, as in `g.names`), one column per function (in the order of `funcs`). |
| `g['w']` | nodes x functions | Weights, the softmax of each row of `z`. Each row sums to 1. Computed from `z`, cannot be set. |
| `g['h1']`, `g['h2']`, ..., `g['o1']`, ... | visible nodes x arguments | One matrix per hidden and output node. |

A node matrix has:

- **One column per argument,** for all functions together. For `funcs=['add', 'mul', 'exp']` there are 2 + 2 + 1 = 5 columns: `add` uses columns 0 and 1, `mul` uses 2 and 3, `exp` uses 4. `g.get_col_names()` returns `['add:1', 'add:2', 'mul:1', 'mul:2', 'exp:1']`.
- **One row per visible node,** from top to bottom: the earlier hidden nodes (nearest first), then the inputs (`x1` last), then the constant row. For `h3` with two inputs the rows are `h2, h1, x2, x1, 1`. For `o1` with three hidden nodes they are `h3, h2, h1, x2, x1, 1`. `g.get_row_names('h3')` returns these names.

So the entry in row `h1`, column `mul:2` of `g['h3']` says how strongly `h1` enters the second argument of `mul` in node `h3`. The last row holds the `c` of each argument.

Parameters can be changed in place (`g['h2'][0, 1] = 0.5`) or replaced (`g['z'] = new_matrix`, the shape must fit).

### Invalid values

All functions of a node are always computed, also those with a weight near zero. If a function returns `nan` or `inf` (for example `log(0)` or `1/0`), that value is replaced by 0.0 before the weighted average is taken. Numpy warnings are switched off inside `get_node2val`.

With random parameters the values can still become very large (but finite), mostly because of `exp` and `power`. A smaller `sigma` or a smaller set of functions helps.

### Reproducing a usual graph

`set_graph` sets all parameters so that the model computes the same as a usual graph:

```python
from get_rand_graph import get_rand_graph

funcs = [('add', 2), ('mul', 2), ('exp', 1)]
names, graph, n_consts = get_rand_graph(n_inp=2, n_con=3, n_hid=3, n_out=1, funcs=funcs)

g = SmoothGraph(n_inp=2, n_hid=3, n_out=1, funcs=funcs)
g.set_graph(names, graph, consts={'c1': 0.5, 'c2': -2.0, 'c3': 1.0})
```

- The graph must have the same number of hidden and output nodes as the model and use only functions of the model.
- `consts` gives a value for every constant used in the graph (`n_consts` of them). They are needed because the smooth graph has no constant nodes: the value goes into the constant row.

What is set for each node:

- In `z`, the function of the node gets the score 50 and all other functions get 0.
- In the node matrix everything is 0, except: an argument that is an input or hidden node gets a 1.0 in the row of that node, and an argument that is a constant gets its value in the last row.

With a score of 50 the weight of the right function is 1 up to about 1e-22, so the other functions still add a tiny amount. In almost all cases you will not see a difference. It matters only if a node sits exactly at a jump of a later function. For example `subtract(x, x)` is exactly 0 in the usual graph and about 1e-22 here, and `power(h, h)` gives 0 for the first and 1 for the second. In a test with 2933 random graphs this happened 6 times.

If you need an exact copy, use `g.set_graph(names, graph, consts, score=np.inf)`. Then the other functions get the score `-inf` and the weights are exactly 1 and 0. In the same test all 2933 graphs were reproduced without any difference. The price is that such scores cannot be changed in small steps afterwards.

Running `python smooth_graph.py` shows a random model and a comparison of a usual graph with its smooth copy.

## Python version

All scripts work with Python 3. Running `python graph.py` runs a few examples: a random graph, small graphs written by hand with backpropagation, evaluation on arrays, and the sequence mode. The random example can print numpy warnings (for example overflow), because random functions are applied to large numbers.
