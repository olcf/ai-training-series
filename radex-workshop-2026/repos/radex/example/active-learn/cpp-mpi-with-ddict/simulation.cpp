#include "radex/dragon.hpp"
#include "radex/handles.hpp"

#include <cstdlib>
#include <mpi.h>
#include <random>
#include <stdexcept>

constexpr double pi = 3.14159265358979323846;
static timespec TIMEOUT = {5, 0};

static size_t N_SAMPLES_PER_RANK = 8;

int main(int argc, char **argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char *ddict_descriptor = getenv("ROSE_DDICT_DESCRIPTOR");
    if (ddict_descriptor == nullptr) {
        throw std::runtime_error(
            "DDict descriptor not found. Ensure that the environment variable "
            "'ROSE_DDICT_DESCRIPTOR' is set.");
    }

    // Create the connection to the existing DDict
    radex::drg::ddict::Client client{ddict_descriptor, &TIMEOUT};

    // Retrieve the iteration count of the active learning loop
    const std::string iter_key("sim_meta_iter_count");
    int iteration = 0;
    if (client.contains(iter_key)) {
        iteration =
            client.get_scalar<int>(radex::data::IncomingHandle{iter_key});
    }
    int previous_iteration = iteration - 1;

    // Initialize the random number generators
    unsigned seed = world_rank + iteration * world_size;
    std::default_random_engine rng(seed);
    std::normal_distribution<double> noise_dist(0.0, 0.1);
    std::uniform_real_distribution<double> query_dist(0.0, 2 * pi);

    // Check to see if we should initialize with random points, or use points
    // from the active learner
    std::vector<double> X_local;
    if (previous_iteration >= 0) {
        std::string query_key("query_points_iter_" +
                              std::to_string(previous_iteration));
        if (client.contains(query_key)) {
            auto [dims, data] = client.get_tensor<double>(
                radex::data::IncomingHandle{query_key});
            X_local = data;
        } else {
            throw std::runtime_error("Query points not found in DDict.");
        }
    }
    for (size_t i = 0; i < N_SAMPLES_PER_RANK; i++) {
        X_local.push_back(query_dist(rng));
    }

    // Create the simulated "y" vector variable
    std::vector<double> y_local;
    for (size_t i = 0; i < X_local.size(); i++) {
        y_local.push_back(sin(X_local[i]) * sin(5. * X_local[i]) +
                          noise_dist(rng));
    }

    // Store the results in the DDict
    std::string result_key_X("sim_rank_" + std::to_string(world_rank) +
                             "_iter_" + std::to_string(iteration) + "_X");
    std::string result_key_y("sim_rank_" + std::to_string(world_rank) +
                             "_iter_" + std::to_string(iteration) + "_y");

    client.put_tensor(radex::data::OutgoingHandle{result_key_X},
                      {X_local.size()}, X_local);
    client.put_tensor(radex::data::OutgoingHandle{result_key_y},
                      {y_local.size()}, y_local);

    // Increase the loop counter for the number of times the simulation has been
    // called
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        client.put_scalar(radex::data::OutgoingHandle{iter_key}, iteration + 1);
    }
    MPI_Finalize();
}
