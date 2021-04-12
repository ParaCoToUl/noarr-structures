#include <string>
#include <functional>

/**
 * Represents a device that has memory
 * (host cpu or a gpu device)
 */
struct memory_device {
    std::uint8_t device_index = -1;

    memory_device() {
        //
    }

    memory_device(std::uint8_t device_index) {
        this->device_index = device_index;
    }
};

class untyped_ship {
public:
    /**
     * Flag that determines whether the ship is considered full or empty
     */
    bool has_payload = false;

    /**
     * Pointer to the underlying buffer
     */
    void* untyped_buffer = nullptr;

    /**
     * Size of the data buffer in bytes
     */
    std::size_t size;

    /**
     * What device this ship lives on
     */
    memory_device device;

    untyped_ship(
        memory_device device,
        void* existing_buffer,
        std::size_t buffer_size
    ) {
        this->device = device;
        this->untyped_buffer = existing_buffer;
        this->size = buffer_size;
    }

protected:
    // virtual method needed for polymorphism..
    // TODO: implement this class and add some virtual methods
    virtual void foo() = 0;
};

template<typename Structure, typename BufferItem = void>
class ship : public untyped_ship {
public:
    /**
     * The structure of data contained on the ship
     */
    Structure structure;

    /**
     * Pointer to the underlying data buffer
     */
    BufferItem* buffer;

    /**
     * Constructs a new ship from an existing buffer
     */
    ship(
        memory_device device,
        void* existing_buffer,
        std::size_t buffer_size
    ) : untyped_ship(device, existing_buffer, buffer_size) {
        this->buffer = (BufferItem*) existing_buffer;
    }

protected:
    // virtual method needed for polymorphism..
    // TODO: implement this class and add some virtual methods
    void foo() override {};
};

class untyped_dock {
public:

    /**
     * Possible states of the dock
     */
    enum state : unsigned char {
        /**
         * No ship at the dock
         */
        empty = 0,

        /**
         * A ship has arrived but hasn't been processed yet
         */
        arrived = 1,

        /**
         * The ship has been processed and is ready to leave
         */
        processed = 2
    };

    /**
     * Returns the state of the dock
     */
    state get_state() {
        if (this->docked_ship == nullptr)
            return state::empty;
        if (this->ship_processed)
            return state::processed;
        return state::arrived;
    }

    /**
     * Set the target dock, to which processed ships are sent
     */
    void send_processed_ships_to(untyped_dock* target) {
        this->ship_target = target;
    }

    /**
     * Returns a reference to the docked ship
     */
    untyped_ship& get_untyped_ship() {
        if (this->docked_ship == nullptr)
            throw std::runtime_error("Cannot get a ship when none is docked.");

        return *this->docked_ship;
    }

    /**
     * Perform a ship arrival to this dock
     */
    void arrive_ship(untyped_ship* ship) {
        if (this->docked_ship != nullptr)
            throw std::runtime_error("There's a ship already present.");

        // TODO: overload the equality operator?
        if (ship->device.device_index != this->device.device_index)
            throw std::runtime_error("The ship belongs to a different device.");

        this->docked_ship = ship;
        this->ship_processed = false;
    }

    /**
     * The device on which the dock exists
     */
    memory_device device;

    untyped_ship* docked_ship = nullptr;
    
    bool ship_processed = false;

private:
    untyped_dock* ship_target = nullptr;
};

template<typename Structure, typename BufferItem = void>
class dock : public untyped_dock {
public:

    /**
     * Returns a reference to the docked ship
     */
    ship<Structure, BufferItem>& get_ship() {
        return dynamic_cast<ship<Structure, BufferItem>&>(
            this->get_untyped_ship()
        );
    }
};

class harbor {
public:

    /**
     * Called by the scheduler before the pipeline starts running
     */
    void scheduler_start() {
        this->perform_dock_registration();
    }

    /**
     * Called by the scheduler when possible,
     * guaranteed to not be called when another
     * update of this harbor is still running
     * 
     * @param callback Call when the async operation finishes,
     * pass true if data was advanced and false if nothing happened
     */
    void scheduler_update(std::function<void(bool)> callback) {
        // if the data cannot be advanced, return immediately
        if (!this->can_advance())
        {
            callback(false);
            return;
        }

        // if the data can be advanced, do it
        this->advance([&](){
            // data has been advanced
            callback(true);
        });
    }

    /**
     * Called by the scheduler after an update finishes.
     * Guaranteed to run on the scheduler thread.
     */
    void scheduler_post_update(bool data_was_advanced) {
        if (data_was_advanced)
            this->post_advance();

        this->send_ships();
    }

protected:

    /**
     * This function is called before the pipeline starts
     * to register all harbor docks
     */
    virtual void register_docks(std::function<void(untyped_dock*)> register_dock) = 0;

    /**
     * Called to test, whether the advance method can be called
     */
    virtual bool can_advance() = 0;

    /**
     * Called to advance the proggress of data through the harbor
     */
    virtual void advance(std::function<void()> callback) = 0;

    /**
     * Called after the data advancement
     */
    virtual void post_advance() {
        //
    }

private:

    std::vector<untyped_dock*> registered_docks;

    /**
     * Calls the register_docks method
     */
    void perform_dock_registration() {
        this->registered_docks.clear();
        
        this->register_docks([&](untyped_dock* d) {
            this->registered_docks.push_back(d);
        });
    }

    /**
     * Sends ships that are ready to leave
     */
    void send_ships() {
        // TODO ...
    }
};

/**
 * Scheduler manages the runtime of a pipeline
 */
class scheduler {
public:

    /**
     * Registers a harbor to be updated by the scheduler
     */
    void add(harbor* h) {
        this->harbors.push_back(h);
    }

    /**
     * Runs the pipeline until no harbors can be advanced.
     */
    void run() {
        
        // call the start method on each harbor
        for (harbor* h : this->harbors)
            h->scheduler_start();
        
        /*
            A naive scheduler implementation with no parallelism.
            
            TODO: parallelize memory transfers, device computation
                and host computation
         */

        bool some_harbor_was_advanced = true;
        do
        {
            some_harbor_was_advanced = false;

            for (harbor* h : this->harbors)
            {
                bool data_was_advanced;

                this->callback_will_be_called();

                h->scheduler_update([&](bool adv){
                    data_was_advanced = adv;
                    this->callback_was_called();
                });

                this->wait_for_callback();

                h->scheduler_post_update(data_was_advanced);
                
                if (data_was_advanced)
                    some_harbor_was_advanced = true;
            }
        }
        while (some_harbor_was_advanced);
    }

private:

    /**
     * Harbors that the scheduler periodically updates
     */
    std::vector<harbor*> harbors;

    ///////////////////////////
    // Synchronization logic //
    ///////////////////////////

    // This is a dummy implementation that only supports synchronous harbors.
    // TODO: Add a proper synchronization primitive on which the scheduler
    // thread can wait and let the callback_was_called method (or its
    // quivalent) be callable from any thread.

    bool _expecting_callback;
    bool _callback_was_called;

    /**
     * Call this before starting a harbor update
     */
    void callback_will_be_called() {
        if (this->_expecting_callback)
            throw std::runtime_error(
                "Cannot expect a callback when the previous didn't finish."
            );

        this->_expecting_callback = true;
        this->_callback_was_called = false;
    }

    /**
     * Call this from the harbor callback
     */
    void callback_was_called() {
        this->_callback_was_called = true;
    }

    /**
     * Call this to synchronously wait for the callback
     */
    bool wait_for_callback() {
        if (!this->_expecting_callback)
            throw std::runtime_error(
                "Cannot wait for callback without first expecting it."
            );

        if (!this->_callback_was_called)
            throw std::runtime_error(
                "TODO: Asynchronous harbors are not implemented yet."
            );

        this->_expecting_callback = false;
    }

};

class my_producing_harbor : public harbor {
public:
    dock<std::size_t, char> output_dock;

    my_producing_harbor(std::string data, std::size_t chunk_size) {
        this->data = data;
        this->chunk_size = chunk_size;
        this->at_index = 0;
    }

    virtual void register_docks(std::function<void(untyped_dock*)> register_dock) {
        register_dock(&this->output_dock);
    };

    bool can_advance() override {
        // false, if we've processed the entire dataset
        if (this->at_index >= this->data.length())
            return false;
        
        // true, if we have an empty ship available
        return this->output_dock.get_state() == untyped_dock::state::arrived;
    }

    void advance(std::function<void()> callback) override {
        // get the ship to be filled up
        auto& ship = this->output_dock.get_ship();

        // compute the size of the next chunk
        std::size_t items_to_take = std::min(
            this->chunk_size,
            this->data.length() - this->at_index
        );

        // move the chunk onto the ship
        this->data.copy(ship.buffer, items_to_take, this->at_index);
        ship.structure = items_to_take;
        ship.has_payload = true;
        this->output_dock.ship_processed = true;

        // update our state
        this->at_index += items_to_take;

        // computation is done
        callback();
    }

private:
    std::string data;
    std::size_t chunk_size;
    std::size_t at_index;
};
