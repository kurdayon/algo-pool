# graph

Small tools for working with math functions written as graphs (DAGs, directed acyclic graphs).

There are two scripts:

- `graph.py` has the `Graph` class. It takes a graph, computes its values for given inputs, and can run backpropagation to get gradients. It can also run the same graph step by step over a sequence, like a recurrent network.
- `get_rand_graph.py` has the `get_rand_graph` function. It builds a random graph in the format that `Graph` understands.

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

| Name | Arguments | What it computes | Gradient |
|---|---|---|---|
| `add` | 2 | `x1 + x2` | yes |
| `mul` | 2 | `x1 * x2` | yes |
| `max` | 2 | larger of `x1`, `x2` | yes (see known issues) |
| `min` | 2 | smaller of `x1`, `x2` | yes (see known issues) |
| `exp` | 1 | `exp(x1)` | yes (see known issues) |
| `subtract` | 2 | `x1 - x2` | no |
| `divide` | 2 | `x1 / x2` | no |
| `power` | 2 | `sign(x1) * abs(x1) ** x2` | no |
| `log` | 1 | `sign(x1) * log(abs(x1))` | no |
| `tanh` | 1 | `tanh(x1)` | no |
| `smif` | 3 | soft "if": `x1 * (1 - w) + x2 * w` with `w = (tanh(x2) + 1) / 2` | no |

`power` and `log` are changed versions that also accept negative numbers.

To add your own function, write a small class with `get_value(...)` (and `get_grads(...)` if you need gradients, returning a list with one derivative per argument) and register it in `fname2func` inside `Graph.__init__`.

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
- `calc_grads=True` works only if every function in the graph has a gradient (see the table). Otherwise you get an `AttributeError`.

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

`seq_eval` returns one dictionary of values per step and one gradient list per step. It always computes gradients, so the graph may only use functions that have a gradient.

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

Running `python get_rand_graph.py` prints one example graph. Note that this built-in example uses the name `plus`, which `Graph` does not know. Use `add` when you want to evaluate the graph.

## Known issues

- **`graph.py` does not load in Python 3.** The example block at the bottom of the file uses Python 2 `print` statements, and Python 3 refuses the whole file because of that, including `from graph import Graph`. Use Python 2, or change those lines to `print(...)`.
- The comment at the top of `graph.py` calls the method `g.eval(...)` and uses the name `multiply`. The real names are `g.get_node2val(...)` and `mul`.
- Gradients of `max` and `min` with respect to the second argument are wrong (they are the same as for the first argument).
- The gradient of `exp` is not returned as a list. With a plain number as input, `calc_grads=True` fails. With an array as input, `back` uses only the first element of the derivative.
- `smif` ignores its third argument.
- The examples at the bottom of `graph.py` stop after the first one because of an `exit()` call.
