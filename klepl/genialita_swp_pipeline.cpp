struct connection1 {
    char* buffer;
    callable producer; // on whichever device and is blocking
};

struct connection2 {
    char* buffer[2];
    callable producer; // on whichever device
};

struct connection3 { // to tady je kvuli mirkove inplace funkci...
    char* buffer[3];
    callable producer; // on whichever device
};

struct map : callable {
    std::array<connection /*connection1 or connection 2 or connection3*/> connections;

    void operator() (vystupni buffery /*vetsinou jeden*/) {
        // call producers
        for (c : connections)
            c.producer(c.buffer[last]); // pseudocode

        // call kernels (pipeline)
        <<<grid>>>kernel(/*vstupni_buffery budou odvozeny z connections (for (c : connections) vstupni_buffery[i++] = *c.buffer -->*/ vstupni_buffery, /*vystupni*/ buffery)
        
        synchronisation_point;

        for (c : connections) // tenhle bude constexpr a cely se to udela chytre
            c.switch_buffers(); // kdyz je to varianta connection1 tak to neudela nic, jinak shiftne (u connections2 proste swap)
    }
};

struct reduce : callable {
    std::array<connection /*connection1 or connection 2*/> connections;


    void operator() (vystupni akumulacni buffer) {
        do {
            // call producers
            for (c : connections)
                c.producer(c.buffer[last]); // pseudocode

            // call kernels (pipeline)
            <<<grid>>>kernel(/*vstupni_buffery budou odvozeny z connections (for (c : connections) vstupni_buffery[i++] = *c.buffer -->*/ vstupni_buffery, /*vystupni*/ buffery)
            
            synchronisation_point;

            for (c : connections) // tenhle bude constexpr a cely se to udela chytre
                c.switch_buffers(); // kdyz je to varianta connection1 tak to neudela nic, jinak shiftne (u connections2 proste swap)
        } while(makes_sense);
    }
};
