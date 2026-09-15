#include <ctime>
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "radex/client.hpp"
#include "radex/dragon.hpp"

using namespace std::literals::chrono_literals;

const std::string IDENT{"               "};

template <typename T> std::string vec_to_str(const std::vector<T> &vec) {
    std::string s{"[ "};
    for (const auto &v : vec) {
        s += std::to_string(v) + ", ";
    }
    s += "]";
    return s;
}

template <typename T>
void print_scalar_key(radex::IClient &client, const std::string &key) {
    std::cout << IDENT << "App: Waiting for scalar key `" << key << "`"
              << std::endl;
    auto val =
        client.wait_for_scalar<T>(radex::data::IncomingHandle{key}, 10'000ms);
    std::cout << IDENT << "App: Got key `" << key << "` has value " << val
              << "\n"
              << std::endl;
}

template <typename T>
void print_vector_key(radex::IClient &client, const std::string &key) {
    std::cout << IDENT << "App: Waiting for tensor key `" << key << "`"
              << std::endl;
    auto [dims, data] =
        client.wait_for_tensor<T>(radex::data::IncomingHandle{key}, 10'000ms);
    std::cout << IDENT << "App: Got key `" << key << "`\n"
              << IDENT << "      |- Data: " << vec_to_str(data) << "\n"
              << IDENT << "      \\- Dims: " << vec_to_str(dims) << "\n"
              << std::endl;
}

int main() {
    const auto DELAY_SET_TIME = 1'000ms;
    char *serialized_dd = getenv("SERIALIZED_DDICT");
    if (serialized_dd == nullptr) {
        throw std::runtime_error("DDict descriptor not found!");
    }

    timespec timeout{5, 0};
    std::cout << IDENT << "App: Creating client" << std::endl;
    radex::drg::ddict::Client client{serialized_dd, &timeout};
    std::cout << IDENT << "App: Client created" << std::endl;

    print_scalar_key<int>(client, "py-int");
    print_scalar_key<double>(client, "py-double");
    print_scalar_key<float>(client, "py-np-float");
    print_vector_key<int>(client, "py-int-tensor");
    print_vector_key<double>(client, "py-float-tensor");

    std::this_thread::sleep_for(DELAY_SET_TIME);
    std::cout << IDENT << "App: Setting Double" << std::endl;
    client.put_scalar<double>(radex::data::OutgoingHandle{"cpp-double"}, 1.23);

    std::this_thread::sleep_for(DELAY_SET_TIME);
    std::cout << IDENT << "App: Setting Int" << std::endl;
    client.put_scalar<int>(radex::data::OutgoingHandle{"cpp-int"}, 987);

    std::this_thread::sleep_for(DELAY_SET_TIME);
    std::cout << IDENT << "App: Setting Double Tensor" << std::endl;
    std::vector<double> v(12);
    std::iota(v.begin(), v.end(), 0);
    client.put_tensor(radex::data::OutgoingHandle{"cpp-double-tensor"}, {4, 3},
                      v);

    std::this_thread::sleep_for(DELAY_SET_TIME);
    std::cout << IDENT << "App: Setting Long Tensor" << std::endl;
    client.put_tensor<int32_t>(radex::data::OutgoingHandle{"cpp-long-tensor"},
                               {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});

    return 0;
}
