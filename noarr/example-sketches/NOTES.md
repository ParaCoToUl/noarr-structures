# Notes


## How envelope hub works - in detail

A hub contains internally a queue of chunks. A *chunk* is a logical piece of data, located on one or more devices (one or more envelopes). A chunk can only enter the queue by being *produced* by a producing link. Therefore the order of chunks in the queue specifies their order in the logical chunk stream and it is what allows "double-buffering" (pipelining). Chunks at the end of the queue may be *consumed* and this is how they leave the queue. Chunks in the middle of the queue may be *peeked* or *modified*. Chunk may not be consumed when it's still being peeked or modified. Modifying a chunk on one device invalidates (frees up) envelopes of that chunk on other devices.

The hub always tries to satisfy all the links that point to it. Producing links may be satistied
whenever there are empty envelopes available. Peeking, modifying and consuming links are satisfied only if the active dataflow strategy lists them (anotherwords, the user has to specify, which link(s) they want to satisfy now). There may be multiple modifying links in the strategy, but they all have to be on the same device and it's the user's responsibility to prevent race-conditions. There may be multiple peeking links on (possibly) multiple devices. There may even be multiple consuming links, but it's advised for them to be manually comitted and the user has to handle race-conditions. A chunk may not be consumed if it's still being modified (if the strategy was changed, but operations haven't finished yet), but it can be consumed when it's still peeked or consumed. The hub keeps a list of consumed but still used chunks and deletes those chunks only when all operations on them finish.

Producing and consuming links have the option of being manually committed. That is, they may not produce/consume a chunk, even though the corresponding comupte node finished. It is needed for production (say, the bitcoin example) and it is needed for consumtion (say, the sobel example). Similarly the production and consumption of chunks may be performed manually, by calling corresponding hub methods.

Dataflow strategy can sometimes be implicitly infered from links.
