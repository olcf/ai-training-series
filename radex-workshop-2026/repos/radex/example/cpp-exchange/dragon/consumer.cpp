#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "radex/dragon.hpp"
#include "radex/handles.hpp"

template <typename T> std::string vec_to_str(const std::vector<T> &vec) {
    std::string s{"[ "};
    for (const auto &v : vec) {
        s += std::to_string(v) + ", ";
    }
    s += "]";
    return s;
}

int main() {

    std::cout << "===========================\n"
              << "Hello World from Consumer!!\n"
              << "---------------------------\n";

    radex::drg::ddict::Client client{};

    auto some_int =
        client.get_scalar<int32_t>(radex::data::IncomingHandle{"some-int"});
    auto [some_int_tensor_dims, some_int_tensor] = client.get_tensor<int32_t>(
        radex::data::IncomingHandle{"some-int-tensor"});

    auto some_float =
        client.get_scalar<double>(radex::data::IncomingHandle{"some-float"});
    auto [some_float_tensor_dims, some_float_tensor] =
        client.get_tensor<double>(
            radex::data::IncomingHandle{"some-float-tensor"});

    std::cout << ""
              << "Some Int:          " << some_int << "\n"
              << "Some Float:        " << some_float << "\n"
              << "Some Int Tensor:   " << vec_to_str(some_int_tensor) << "\n"
              << "      \\ -> Dims:   " << vec_to_str(some_int_tensor_dims)
              << "\n"
              << "Some Float Tensor: " << vec_to_str(some_float_tensor) << "\n"
              << "        \\ -> Dims: " << vec_to_str(some_float_tensor_dims)
              << "\n"
              << "---------------------------\n"
              << "Goodbye from Consumer!!\n"
              << "===========================\n";

    return 0;
}
