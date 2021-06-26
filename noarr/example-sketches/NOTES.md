# What I noticed

- The `initializer` compute node repeats. It just puts data into hubs once.
    Why not do it as part of the hub initialization process?
    ...Same thing seems to happen at the end, when we just want to pull the value
    out of a hub after it has been computed (`finalizer`).
    ...Add helper methods onto the pipeline object that are called before
    pipeline starts and after pipeline ends

- Sending envelopes by scheduler is maybe not the right approach. A compute node
    seems to "peek into a hub and look at an envelope" instead of receiving it
    and sending back. At least that's how concurrent access seems to work.

- Also, compute node may not produce a chunk of data -- keep thinking about
    chunk streams.

- Who will manage hub and compute node lifecycle?
    ...Define a pipeline object that will act as a factory of these things
    and will contain a scheduler
    and will make sure no compute node is deleted when it shouldn't

- Maybe we want two nodes to produce one chunk in a shared hub (shared write)
    ...Support it in a similar way to shared read

- Moving envelopes between hubs MAKES EVERYTHING WAY MORE COMPLICATED
    ...Instead let's allow two envelopes of the same size and device
    "swap contents" by swapping their internal pointers. That will do the trick.

- An envelope more and more looks like the original *bag* idea. And hub looks
    more and more like the original *envelope* idea.

- Dataflow strategy could be implicitly infered in many cases (e.g. one read, one write link)

- can_advance(return true) is present often, should be a default when omitted.


# ENVELOPE HUB - how it works?

Queue of chunks. Link looks at a chunk (an envelope specifically). The chunk may just be being created, consumed, modified, or read (=peeked). Some links always produce/consume chunks. Some only peek/modify. If a link creates/consumes only sometimes, the consumption has be done manually by a function call during the node execution.

Dataflow strategy tells the hub, how to transfer chunks between devices. It's basically a set of devices where the chunks need to be present.

Write can happen only to a chunk that resides on one device (since we don't know how to merge changes).
