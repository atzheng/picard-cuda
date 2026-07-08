#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <vector>
#include <fstream>

#include "types.h"
#include "data_loader.h"
#include "algorithm.h"

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s <input_dir> [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --max_workers N          (default: 1)\n");
    fprintf(stderr, "  --max_products_per_worker N (default: 1000000)\n");
    fprintf(stderr, "  --num_steps N            (default: 10)\n");
    fprintf(stderr, "  --max_iters N            (default: 10000000)\n");
    fprintf(stderr, "  --seed N                 (default: 42)\n");
    fprintf(stderr, "  --greedy_cost_prob F     (default: 1.0)\n");
    fprintf(stderr, "  --layer_sizes N,N,N,...  (default: 30,64,64,31)\n");
    fprintf(stderr, "  --output FILE            CSV output for timing\n");
    fprintf(stderr, "  --fulfill_output FILE    .npy output for fulfillment\n");
    fprintf(stderr, "  --mode MODE              'picard' (default), 'sequential', or 'benchmark'\n");
}

static std::vector<int> parse_layer_sizes(const char* str) {
    std::vector<int> sizes;
    std::string s(str);
    size_t pos = 0;
    while (pos < s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) end = s.size();
        sizes.push_back(std::stoi(s.substr(pos, end - pos)));
        pos = end + 1;
    }
    return sizes;
}

// Write a 1D int array as .npy
static void write_npy_int(const std::string& path, const int* data, int n) {
    std::ofstream f(path, std::ios::binary);
    // Magic
    f.write("\x93NUMPY", 6);
    // Version 1.0
    uint8_t major = 1, minor = 0;
    f.write(reinterpret_cast<char*>(&major), 1);
    f.write(reinterpret_cast<char*>(&minor), 1);

    char header[256];
    int header_len = snprintf(header, sizeof(header),
        "{'descr': '<i4', 'fortran_order': False, 'shape': (%d,), }", n);
    // Pad to multiple of 64
    int total = 10 + header_len + 1;  // 10 = magic(6) + version(2) + header_len(2)
    int pad = ((total + 63) / 64) * 64 - total;
    header_len += pad + 1;  // +1 for newline

    uint16_t hl = (uint16_t)header_len;
    f.write(reinterpret_cast<char*>(&hl), 2);

    // Write header dict
    int written = snprintf(header, sizeof(header),
        "{'descr': '<i4', 'fortran_order': False, 'shape': (%d,), }", n);
    f.write(header, written);
    for (int i = 0; i < header_len - written - 1; i++) f.put(' ');
    f.put('\n');

    // Write data
    f.write(reinterpret_cast<const char*>(data), n * sizeof(int));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_dir = argv[1];

    AlgoConfig config;
    config.max_workers = 1;
    config.max_products_per_worker = 1000000;
    config.num_steps = 10;
    config.max_iters = 10000000;
    config.layer_sizes = {30, 64, 64, 31};
    config.greedy_cost_prob = 1.0;
    config.seed = 42;

    std::string output;
    std::string fulfill_output;
    std::string mode = "picard";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--max_workers") == 0 && i + 1 < argc) {
            config.max_workers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max_products_per_worker") == 0 && i + 1 < argc) {
            config.max_products_per_worker = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--num_steps") == 0 && i + 1 < argc) {
            config.num_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max_iters") == 0 && i + 1 < argc) {
            config.max_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            config.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--greedy_cost_prob") == 0 && i + 1 < argc) {
            config.greedy_cost_prob = atof(argv[++i]);
        } else if (strcmp(argv[i], "--layer_sizes") == 0 && i + 1 < argc) {
            config.layer_sizes = parse_layer_sizes(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--fulfill_output") == 0 && i + 1 < argc) {
            fulfill_output = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Load problem data
    printf("Loading problem from %s...\n", input_dir.c_str());
    ProblemData problem = load_problem(input_dir, 0);

    // Validate config
    if (config.max_workers * config.max_products_per_worker < problem.n_products) {
        fprintf(stderr, "Error: max_workers * max_products_per_worker (%d) < n_products (%d)\n",
                config.max_workers * config.max_products_per_worker, problem.n_products);
        return 1;
    }

    // Adjust first layer size to match n_nodes if needed
    if (config.layer_sizes[0] != problem.n_nodes) {
        printf("Adjusting NN input size from %d to %d (n_nodes)\n",
               config.layer_sizes[0], problem.n_nodes);
        config.layer_sizes[0] = problem.n_nodes;
    }
    // Adjust last layer size to match n_nodes+1 if needed
    int np1 = problem.n_nodes + 1;
    if (config.layer_sizes.back() != np1) {
        printf("Adjusting NN output size from %d to %d (n_nodes+1)\n",
               config.layer_sizes.back(), np1);
        config.layer_sizes.back() = np1;
    }

    if (mode != "picard" && mode != "sequential" && mode != "benchmark") {
        fprintf(stderr, "Error: --mode must be 'picard', 'sequential', or 'benchmark', got '%s'\n", mode.c_str());
        return 1;
    }

    // Create GPU structures
    printf("Initializing GPU state...\n");
    NeuralNet nn = create_neural_net(config, (uint64_t)config.seed);
    Event events = create_events(problem);
    AlgoState state = create_algo_state(problem, config, (uint64_t)config.seed);

    if (mode == "benchmark") {
        int bench_steps = std::min(config.num_steps, problem.n_events);
        run_benchmark(state, events, nn, bench_steps);

        free_algo_state(state);
        free_events(events);
        free_neural_net(nn);
        return 0;
    }

    // Run
    auto t0 = std::chrono::high_resolution_clock::now();

    if (mode == "sequential") {
        printf("--- Running sequential (CUDA, single thread)... ---\n");
        run_sequential_day(state, events, nn, config);
    } else {
        printf("--- Running Picard iteration (CUDA, %d workers)... ---\n", config.max_workers);
        run_algorithm_day(state, events, nn, config);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (mode == "picard") {
        printf("Iterations: %d, t_reset: %d/%d, conflicts: %d\n",
               state.iteration, state.t_reset, state.n_events, state.n_conflicts);
    } else {
        printf("Processed: %d/%d events\n", state.t_reset, state.n_events);
    }
    printf("--- total time: %.3f seconds ---\n", elapsed);

    // Copy results
    std::vector<int> h_fulfill(problem.n_events);
    get_fulfill(state, h_fulfill.data());

    // Convert sentinel to -1
    for (int i = 0; i < problem.n_events; i++) {
        if (h_fulfill[i] == problem.n_nodes) h_fulfill[i] = -1;
    }

    // Write outputs
    if (!output.empty()) {
        FILE* f = fopen(output.c_str(), "w");
        fprintf(f, "compile_time,run_time_1,run_time_2,conflicts,iterations_1,iterations_2\n");
        fprintf(f, "0.0,%.6f,%.6f,%d,%d,%d\n",
                elapsed, elapsed, state.n_conflicts, state.iteration, state.iteration);
        fclose(f);
        printf("Wrote timing to %s\n", output.c_str());
    }

    if (!fulfill_output.empty()) {
        write_npy_int(fulfill_output, h_fulfill.data(), problem.n_events);
        printf("Wrote fulfillment to %s\n", fulfill_output.c_str());
    }

    // Cleanup
    free_algo_state(state);
    free_events(events);
    free_neural_net(nn);

    return 0;
}
