#include <cstddef>
#include "noarr/pipelines.hpp"

using namespace noarr::pipelines;

class my_producing_node : public producer_compute_node<std::size_t, int> {
private:
    std::vector<int>* items;
    std::size_t at;

public:
    my_producing_node(std::vector<int>* items) {
        this->items = items;
        this->at = 0;
    }

    bool is_ready_for_next_chunk() override {
        return ! this->output_port.contains_chunk();
    }

    void start_next_chunk_processing() override {
        int* output_buffer = this->output_port.get_buffer();
        std::size_t at_before = this->at;
        for (int i = 0; i < 3 && this->at < this->items->size(); i++) {
            output_buffer[i] = (*this->items)[this->at];
            this->at++;
        }
        this->output_port.set_structure(this->at - at_before);

        // TODO: maybe also distinguish "being filled up / being consumed" states?
        this->output_port.set_contains_chunk(true);

        if (this->at == this->items->size() - 1) {
            this->output_port.send_end_of_stream();
            this->set_all_chunks_processed();
        }

        this->set_chunk_processing_finished();
    }
};

class my_mapping_node : public pipe_compute_node<
    std::size_t, std::size_t, int, int
> {

    bool is_ready_for_next_chunk() override {
        return this->input_port.contains_chunk()
            && !this->output_port.contains_chunk();
    }

    void start_next_chunk_processing() override {
        // === check end of stream ===
        if (this->input_port.contains_end_of_stream())
        {
            this->set_all_chunks_processed(); // we're done as a node
            this->output_port.send_end_of_stream(); // pass EOS downstream
            return;
        }

        // === perform the mapping operation ===

        // NOTE/TODO: here will be cast to a "bag" type
        std::size_t item_count = this->input_port.get_structure();
        int* input_buffer = this->input_port.get_buffer();
        int* output_buffer = this->output_port.get_buffer();

        // copy the number of items
        this->output_port.set_structure(item_count);

        // perform the map operation on the values
        for (int i = 0; i < item_count; i++)
            output_buffer[1 + i] = input_buffer[1 + i] * 2; // map = *2

        // the input buffer was consumed and the output buffer
        // was filled with a chunk of data
        this->input_port.set_contains_chunk(false);
        this->output_port.set_contains_chunk(true);

        // the asynchronous operation has finished
        // (well, it wasn't asnychronous at all in this case)
        this->set_chunk_processing_finished();
    }
};

class my_printing_node : public consumer_compute_node<std::size_t, int> {
public:
    std::string log;

    my_printing_node() {
        log.clear();
    }

    bool is_ready_for_next_chunk() override {
        return this->input_port.contains_chunk();
    }

    // TODO: separate external and internal API!
    void start_next_chunk_processing() override {
        // === check end of stream ===
        if (this->input_port.contains_end_of_stream())
        {
            this->set_all_chunks_processed();
            return;
        }

        // process the input chunk
        std::size_t item_count = this->input_port.get_structure();
        int* input_buffer = this->input_port.get_buffer();

        for (std::size_t i = 0; i < item_count; i++) {
            this->log.append(
                std::to_string(input_buffer[i])
            );
            this->log.append(";");
        }

        this->input_port.set_contains_chunk(false);

        this->set_chunk_processing_finished();
    }
};

// TOHLE CHCE UŽIVATEL - TOHLE JE LAMBDA
class my_node : compute_node {
    auto lmbd = [](
        link<std::size_t, int, input> input,
        link<std::size_t, int, input> input2,
        link<std::size_t, float, output> output
    ) {
        int* input_buffer = input->get_buffer();
        int* input_buffer2 = input2->get_buffer();
        float* output_buffer = output->get_buffer();

        std::size_t items_count_1 = input->get_structure();

        for (std::size_t i = 0; i < items_count_1; i++)
            output_buffer[i] = input_buffer[i] * 2;
    };
}

void my_pipeline_running_function() {
    // prepare data that will go through the pipeline
    std::vector<int> items {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    // build the pipeline
    auto prod = my_producing_node(&items);

    auto env = move_h2d_envelope<std::size_t, int>();
    prod.set_output_port(env.get_input_port());
    prod.attach_envelope(&env, envelope::WRITE_ATTACHMENT);

    auto print = my_printing_node();
    print.attach_envelope(&env, envelope::READ_ATTACHMENT);
    print.set_input_port(env.get_output_port());

    // run the pipeline to completion
    while (!print.has_processed_all_chunks())
    {
        if (prod.is_ready_for_next_chunk())
            prod.start_next_chunk_processing();

        if (env.is_ready_for_next_chunk())
            env.start_next_chunk_processing();

        if (print.is_ready_for_next_chunk())
            print.start_next_chunk_processing();

        // sleep_on_some_lock()
    }

    // print the result
    // print.log;
}

void my_better_pipeline_building_approach() {
    // prepare data that will go through the pipeline
    std::vector<int> items {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    // build the pipeline (manual)
    auto prod = my_producing_node(&items);

    auto env = move_h2d_envelope<std::size_t, int>();
    prod.set_output_port(env.get_input_port());

    auto print = my_printing_node();
    print.set_input_port(env.get_output_port());

    // build the pipeline
    /*linear_pipeline pip =*/ linear_pipeline::builder()
        .foo<my_producing_node>();
        //.start_node<my_producing_node, std::size_t, int>(&items)
        // .start_node_new(
        //     static_cast<std::unique_ptr<producer_compute_node<std::size_t, int>>>(
        //         std::make_unique<my_producing_node>(&items)
        //     )
        // )
        //.start_node_existing(&prod)
        //.envelope_existing(&env);

    auto pipeline = linear_pipeline::builder()
        .start_node_existing(&prod)
        .follows(&env) // secondary API, but pass only a poitner and don't start owning!
        .ends_with(&print);

    // Primary API:
    auto pipeline = linear_pipeline::builder()
        .start_node<my_producing_node>(&items)
        .envelope<move_h2d_envelope<std::size_t, int>>()
        .node<>
        .end_node<my_printing_node>();

    //std::make_unique<my_producing_node>(&items);

    // run the pipeline to completion
    pipeline.run();

    // print the result
    // print.log;
}


///////////////////
// Rewrite 29.3. //
////////////////////////////////////////////////////////////////////////////////

/*
    Envelope
    --------

    Envelope logicky představuje jeden buffer, do kterého mají přístup
    compute nody. Jeden takový přístup se nazývá link. Link má následující
    vlastnosti:
        - accessMode: R, W, RW
        - residence: host, device
    V ideálním světě je envelope "nějaký buffer" a čtení/zápis z libovolného
    linku edituje jeho obsah. V praxi musíme řešit pohyb dat nebo swapování
    bufferů, proto je potřeba zavést vlastnosti linků a nechávat envelope na
    základě techto vlastností data distribuovat. Můžeme si to představit tak,
    v každém linku je jedna kopie dat a veškeré kopie v envelope se musí
    udržovat synchronizované. K desynchronizaci může dojít, pokud nějaký
    link typu W nebo RW sáhne na data (touch). V takový okamžik mají najednou
    všechny ostatní linky zastaralá data a je třeba provést synchronizaci.
    Odsud má každý link oepraci `touch` a každá envelope operaci `synchronize`.
        NOTE: Výhoda R linků je, že nezpůsobují desynchronizaci.
    ~~~~~~~~~~~ Odsud je to vymyšlené jen napůl:
    Ještě je třeba spolu s každým bufferem svázat nějaká další data:
        - hodnota: structure
        - příznaky: has EOS, has chunk, vlastní (62 dalších)
    Structure popisuje strukturu dat v bufferu. EOS je signál, který
    propaguje skrz pipeline informaci o konci výpočtu. Has chunk je příznak,
    který značí, zda buffer obsahuje zajímavá data, nebo je prázdný.
    Příznaky nespouští operaci `touch`, jelikož jsou uložené jednou,
    pro celou envelope a tedy mohou být nastavovány i linky typu R.
    ~~~~~~~~~~~ Ještě optimalizace:
    Operace `synchronize` musí vědět, do jakých linků chceme data dostat,
    aby se např. nepřesovaly na hosta, není-li to třeba. Čili compute node
    se může spustit, když všechny jeho linky jsou synchronizované, ne až když
    celá envelopa je synchronizovaná.
    Ještě note: nechci jen vlastní flagy, ale obecně libovolná metadata. Někdy
    může mít compute node dva vstupy a chci aby byly oba ze stejného chunku
    (e.g. v H.265 mám pipeline takovou strukturu)


    Compute node
    ------------

    Compute node je cokoliv, co provádí výpočet, a pro ten to vyžaduje odkazy
    na envelopy. Když chci compute node spustit, tak vím jaké má R a RW linky
    a musím tedy zajistit, aby se příslušné envelopy synchronizovaly, jinak
    uzel nemůžu spustit.


    Scheduler
    ---------

    Scheduler strká do uzlů aby běžely. Každý uzel umí poznat, kdy může běžet.
    Scheduler doběhne, když není žádný uzel, který by se nechal pošťouchnout.
    Každý uzel vidí jen na připojené envelopy, čili komunikace o stavu výpočtu
    musí probíhat skrz příznaky envelop. Někdy je třeba přidat vlastní přínaky,
    pokud je jedna envelopa sdílená více uzly v režimu RW a uzly mají běžet
    nějak sekvenčně po sobě.

    NOTE: Nebude-li se stav výpočtu předávat přes příznaky envelop, tak musí
    být scheduler chytřejší -> uživatel si ho musí napsat sám, ALE to ztěžuje
    paralelizovatelnost (kernel jde paralelizovat třeba s cudaCopy nebo cpu
    výpočtem). Touhle cestou bych se nepouštěl.
 */

void lambda_based_envelope_linking_with_scheduler() {
    /*
        Scenario:
        For each chunk:
            Items get loaded into an envelope `frame_envelope`.
            Then a mapping operation takes place (*2)
            Then a reduction operation takes place (sum) and outputs the
                result into an `aggregation_envelope`
            Then the result is printed.
     */

    // input items
    std::vector<int> items {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 // ...
    };

    // create the pipeline
    auto frame_envelope = envelope<std::size_t, int>();
    bool frame_envelope_was_mapped = false;
    auto aggregation_envelope = envelope<void, int>();

    auto producer = lambda_compute_node()
        .start_when([&](compute_node* node) {
            return items.size() > 0 && !frame_envelope.contains_data;
        })
        .prepare_envelopes([&](compute_node* node) {
            frame_envelope.prepare_buffer(
                envelope::MODE_WRITE, envelope::RESIDENCE_HOST
            );
            // NOTE: (only adds jobs onto some queue and then processes async)
        })
        .computation([&](compute_node* node) {
            // add three more items to the frame buffer
            std::size_t items_to_produce = std::max(3UL, items.size());
            frame_envelope.structure = items_to_produce;
            int* frame = frame_envelope.buffer; // NOTE: set by preparation
            for (int i = 0; i < items_to_produce; i++) {
                frame[i] = *items.end();
                items.pop_back();
            }

            node->computation_done();
        })
        .finalization([&](compute_node* node) {
            // runs on the scheduler thread again,
            // after the callback is called (->computation_done())
            frame_envelope.contains_data = true;
        });

    auto mapper = lambda_compute_node()
        .start_when([&](compute_node* node) {
            return frame_envelope.contains_data && !frame_envelope_was_mapped;
        })
        .prepare_envelopes([&](compute_node* node) {
            frame_envelope.prepare_buffer(
                envelope::MODE_READ_WRITE, envelope::RESIDENCE_DEVICE
            );
            // NOTE: (only adds jobs onto some queue and then processes async)
        })
        .computation([&](compute_node* node) {
            std::size_t items_to_map = frame_envelope.structure;
            int* frame = frame_envelope.buffer; // NOTE: set by preparation
            // for (int i = 0; i < items_to_map; i++) {
            //     frame[i] *= 2;
            // }
            run_mapping_kernel<<<?,?>>>(items_to_map, frame);
            cudaSynchronize();

            frame_envelope_was_mapped = true;
            node->computation_done();
        })
        .finalization([&](compute_node* node) {
            frame_envelope_was_mapped = true;
        });

    // TODO: reducer

    // TODO: printer
    
    // create scheduler
    auto s = scheduler();
    s.add_envelope(frame_envelope);
    s.add_envelope(aggregation_envelope);
    s.add_node(producer);
    s.add_node(mapper);
    s.add_node(reducer);
    s.add_node(printer);

    // run to completion
    s.run();
}
