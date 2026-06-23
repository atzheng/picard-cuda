#include "data_loader.h"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>

// Minimal .npy file reader
// Supports: float64, int64, int32, float32 (converts all to the target type)
// Only handles C-contiguous, little-endian, non-Fortran order arrays.

struct NpyHeader {
    std::string descr;  // e.g. "<f8", "<i8", "<i4", "<f4"
    bool fortran_order;
    std::vector<size_t> shape;
};

static NpyHeader parse_npy_header(std::ifstream& f) {
    // Read magic: \x93NUMPY
    char magic[6];
    f.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        throw std::runtime_error("Not a .npy file");
    }

    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl;
        f.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    NpyHeader h;
    h.fortran_order = false;

    // Parse descr
    auto descr_pos = header.find("'descr'");
    if (descr_pos == std::string::npos) descr_pos = header.find("\"descr\"");
    if (descr_pos != std::string::npos) {
        auto q1 = header.find("'", descr_pos + 7);
        if (q1 == std::string::npos) q1 = header.find("\"", descr_pos + 7);
        auto q2 = header.find_first_of("'\"", q1 + 1);
        h.descr = header.substr(q1 + 1, q2 - q1 - 1);
    }

    // Parse fortran_order
    if (header.find("True") != std::string::npos &&
        header.find("fortran_order") != std::string::npos) {
        auto fo_pos = header.find("fortran_order");
        auto true_pos = header.find("True", fo_pos);
        if (true_pos != std::string::npos && true_pos < fo_pos + 30) {
            h.fortran_order = true;
        }
    }

    // Parse shape
    auto shape_pos = header.find("(");
    auto shape_end = header.find(")", shape_pos);
    std::string shape_str = header.substr(shape_pos + 1, shape_end - shape_pos - 1);
    // Parse comma-separated integers
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
        if (pos >= shape_str.size()) break;
        size_t end = shape_str.find_first_of(", )", pos);
        if (end == std::string::npos) end = shape_str.size();
        h.shape.push_back(std::stoull(shape_str.substr(pos, end - pos)));
        pos = end;
    }

    return h;
}

static std::vector<float> read_npy_as_float(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    NpyHeader h = parse_npy_header(f);

    size_t total = 1;
    for (auto s : h.shape) total *= s;

    std::vector<float> result(total);

    if (h.descr == "<f8" || h.descr == "float64") {
        std::vector<double> buf(total);
        f.read(reinterpret_cast<char*>(buf.data()), total * sizeof(double));
        for (size_t i = 0; i < total; i++) result[i] = (float)buf[i];
    } else if (h.descr == "<f4" || h.descr == "float32") {
        f.read(reinterpret_cast<char*>(result.data()), total * sizeof(float));
    } else if (h.descr == "<i8" || h.descr == "int64") {
        std::vector<int64_t> buf(total);
        f.read(reinterpret_cast<char*>(buf.data()), total * sizeof(int64_t));
        for (size_t i = 0; i < total; i++) result[i] = (float)buf[i];
    } else if (h.descr == "<i4" || h.descr == "int32") {
        std::vector<int32_t> buf(total);
        f.read(reinterpret_cast<char*>(buf.data()), total * sizeof(int32_t));
        for (size_t i = 0; i < total; i++) result[i] = (float)buf[i];
    } else {
        throw std::runtime_error("Unsupported dtype: " + h.descr);
    }

    return result;
}

static std::vector<int> read_npy_as_int(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    NpyHeader h = parse_npy_header(f);

    size_t total = 1;
    for (auto s : h.shape) total *= s;

    std::vector<int> result(total);

    if (h.descr == "<i8" || h.descr == "int64") {
        std::vector<int64_t> buf(total);
        f.read(reinterpret_cast<char*>(buf.data()), total * sizeof(int64_t));
        for (size_t i = 0; i < total; i++) result[i] = (int)buf[i];
    } else if (h.descr == "<i4" || h.descr == "int32") {
        std::vector<int32_t> buf(total);
        f.read(reinterpret_cast<char*>(buf.data()), total * sizeof(int32_t));
        for (size_t i = 0; i < total; i++) result[i] = (int)buf[i];
    } else if (h.descr == "<f8" || h.descr == "float64") {
        std::vector<double> buf(total);
        f.read(reinterpret_cast<char*>(buf.data()), total * sizeof(double));
        for (size_t i = 0; i < total; i++) result[i] = (int)buf[i];
    } else if (h.descr == "<f4" || h.descr == "float32") {
        std::vector<float> buf(total);
        f.read(reinterpret_cast<char*>(buf.data()), total * sizeof(float));
        for (size_t i = 0; i < total; i++) result[i] = (int)buf[i];
    } else {
        throw std::runtime_error("Unsupported dtype: " + h.descr);
    }

    return result;
}

static std::vector<size_t> read_npy_shape(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    NpyHeader h = parse_npy_header(f);
    return h.shape;
}

ProblemData load_problem(const std::string& directory, int day) {
    ProblemData prob;

    std::string sep = "/";

    // Load inventory: [n_products, n_nodes]
    auto inv_shape = read_npy_shape(directory + sep + "inventory.npy");
    prob.n_products = (int)inv_shape[0];
    prob.n_nodes = (int)inv_shape[1];
    prob.inventory = read_npy_as_float(directory + sep + "inventory.npy");

    // Load capacity: [n_days, n_nodes] — extract single day
    auto cap_all = read_npy_as_float(directory + sep + "capacity.npy");
    auto cap_shape = read_npy_shape(directory + sep + "capacity.npy");
    prob.capacity.resize(prob.n_nodes);
    for (int n = 0; n < prob.n_nodes; n++) {
        prob.capacity[n] = cap_all[day * prob.n_nodes + n];
    }

    // Load order_products: [n_events]
    prob.order_products = read_npy_as_int(directory + sep + "order_products.npy");
    prob.n_events = (int)prob.order_products.size();

    // Load node_index_near_to_far: [n_events, n_nodes]
    prob.node_index_near_to_far = read_npy_as_int(directory + sep + "node_index_near_to_far.npy");

    std::cout << "Loaded problem: "
              << prob.n_products << " products, "
              << prob.n_nodes << " nodes, "
              << prob.n_events << " events" << std::endl;

    return prob;
}
