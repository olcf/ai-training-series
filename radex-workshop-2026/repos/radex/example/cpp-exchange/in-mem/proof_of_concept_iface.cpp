#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "radex/client.hpp"
#include "radex/handles.hpp"

namespace pkg {

namespace localkv {

class UnorderedMapClient : public radex::IClient {
  public:
    std::unordered_map<std::string, std::vector<uint8_t>> _map{};

    bool contains(std::string_view key) override {
        try {
            _map.at(std::string{key});
        } catch (std::out_of_range &e) {
            return false;
        }
        return true;
    }

    private:
        void delete_key(std::string_view key) override {
        _map.erase(std::string{key});
    }

    public:
    void put_bytes(std::string_view key, const void *bytes,
                   radex::detail::MetaInt length) override {
        auto ptr = static_cast<const std::uint8_t *>(bytes);
        _map.insert(
            {std::string{key}, std::vector<uint8_t>{ptr, ptr + length}});
    }

    radex::detail::BytesBuffer get_bytes(std::string_view key) override {
        auto buf = _map.at(std::string{key});
        auto ptr = std::make_unique<uint8_t[]>(buf.size());
        std::copy(buf.begin(), buf.end(), ptr.get());
        return {std::move(ptr), buf.size()};
    }

    radex::detail::BytesBuffer
    wait_for_bytes(std::string_view key,
                   std::chrono::milliseconds timeout) override {
        return IClient::wait_for_bytes(key, timeout);
    }
};

} // namespace localkv

} // namespace pkg

template <typename T> std::string vec_to_str(const std::vector<T> &vec) {
    std::string s{"[ "};
    for (const auto &v : vec) {
        s += std::to_string(v) + ", ";
    }
    s += "]";
    return s;
}

int main() {
    int i = 123;
    double d = 1.23;

    pkg::localkv::UnorderedMapClient client{};
    client.put_scalar(radex::data::OutgoingHandle{"some-int"}, i);
    client.put_scalar(radex::data::OutgoingHandle{"some-double"}, d);

    std::cout << "=============\n"
              << "Hello World!!\n"
              << "=============\n";

    auto some_int =
        client.get_scalar<int>(radex::data::IncomingHandle{"some-int"});
    auto some_double =
        client.get_scalar<double>(radex::data::IncomingHandle{"some-double"});

    std::vector<int> itensor{0, 1, 2, 3};
    std::vector<double> dtensor{0.1, 2.3, 4.5, 6.7};

    client.put_tensor(radex::data::OutgoingHandle{"int-tensor"}, {2, 2},
                      itensor);
    client.put_tensor(radex::data::OutgoingHandle{"double-tensor"}, {4},
                      dtensor);

    auto [some_itensor_dims, some_itensor] =
        client.get_tensor<int>(radex::data::IncomingHandle{"int-tensor"});
    auto [some_dtensor_dims, some_dtensor] =
        client.get_tensor<double>(radex::data::IncomingHandle{"double-tensor"});

    std::cout << ""
              << "Some Int:           " << some_int << "\n"
              << "Some Double:        " << some_double << "\n"
              << "\n"
              << "Some Int Tensor:    " << vec_to_str(some_itensor) << "\n"
              << "        \\- Dims:    " << vec_to_str(some_itensor_dims)
              << "\n"
              << "Some Double Tensor: " << vec_to_str(some_dtensor) << "\n"
              << "           \\- Dims: " << vec_to_str(some_dtensor_dims)
              << "\n"
              << "=============\n";
}
