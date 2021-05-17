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
