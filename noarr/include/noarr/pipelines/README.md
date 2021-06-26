# Noarr pipelines

This file is a documentation outline.


## `Node`

Something the scheduler pokes. It can say whether it can advance data when poked.

Node doesn't care about data - where it gets it, how it communicates with other nodes, etc... It only cares about computation.

Compute nodes, hubs and links are higher-level objects that formalize the way nodes communicate with one another.


## `ComputeNode`

A computation node that is meant to use data from hubs.


## `Hub`

A node that is responsible for data management (allocation and transfer).


## `Link`

Something that formalizes data exchange between two nodes. It lets one node look at data owned by another node. The owner of the data is the *host* and the requester is called the *guest*. But that nomenclature is internal, it need not leak to the external API.

There are some flags specifying a link meaning:

- produce+consume / peek
