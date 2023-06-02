


# IMLCV


Welcome to the documentation of IMLCV package.

```{admonition} Warning
:class: warning

This package is work in progress. APIs are subject to change and some features are not yet implemented.
```

## Philosophy

This package is build on 2 ideas. On the one hand, we want to define collective variables in a very flexible way. Collective variables should be composable functions of cell parameters, system coordinates and optinally a neighbourghlist, captured in {class}`IMLCV.base.CV.SystemParams` class and {class}`IMLCV.base.CV.NeighbourList` class.



### Flexible CV definitions and biases
#todo

### Iterative CV determination
#todo



for more details, see [philosophy](philosophy) page.

##  Installation


On a local machine, you can just use


```
pip install numpy cython
pip install .
```

for more details, see [installation](Installation) page.

## Getting started


```
   python IMLCV/examples/simul.py --help
```




```{toctree}
---
caption: Getting started
maxdepth: 2
---


Installation
Getting started
Examples <notebook_link>

Contributions & Help <contributing>
License <license>
Authors <authors>
Changelog <changelog>
Module Reference <api/modules>

```
