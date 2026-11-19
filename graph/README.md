# graph

Small tools for working with math functions written as graphs (DAGs, directed acyclic graphs).

There are three scripts:

- `graph.py` has the `Graph` class. It takes a graph, computes its values for given inputs, and can run backpropagation to get gradients. It can also run the same graph step by step over a sequence, like a recurrent network.
- `get_rand_graph.py` has the `get_rand_graph` function. It builds a random graph in the format that `Graph` understands.
- `mutate_graph.py` has the `mutate_graph` function. It takes a graph and returns a slightly changed (mutated) graph that is still valid.

The only dependency is `numpy` (needed by `graph.py`).

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

What you can rely on:

- The result can be given directly to `Graph(names, graph)`.
- Every hidden node is used by a later hidden node or by an output.
- No node is computed from constants only, so every node depends on at least one input.
- `n_consts` tells how many constants the graph really uses. It can be smaller than `n_con`. The used constants are always named `c1 ... c<n_consts>`, so you know exactly which values to provide.
- It is not guaranteed that every input is used.

Running `python get_rand_graph.py` prints one example graph.

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

## Python version

All scripts work with Python 3. Running `python graph.py` runs a few examples: a random graph, small graphs written by hand with backpropagation, evaluation on arrays, and the sequence mode. The random example can print numpy warnings (for example overflow), because random functions are applied to large numbers.
