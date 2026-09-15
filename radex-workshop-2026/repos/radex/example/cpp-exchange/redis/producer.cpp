#include <ctime>
#include <iostream>

#include "radex/handles.hpp"
#include "radex/smartredis.hpp"

int main() {
    std::cout << "===========================\n"
              << "Hello World from Producer!!\n"
              << "---------------------------\n";

    radex::redis::smartredis::Client client{"example-sr-producer"};

    client.put_scalar<int32_t>(radex::data::OutgoingHandle{"some-int"}, 123);
    client.put_scalar<double>(radex::data::OutgoingHandle{"some-float"}, 1.23);

    client.put_tensor<double>(radex::data::OutgoingHandle{"some-float-tensor"},
                              {4}, {0.12, 3.45, 6.78, 9.123});
    client.put_tensor<int32_t>(radex::data::OutgoingHandle{"some-int-tensor"},
                               {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});

    std::cout << "---------------------------\n"
              << "Goodbye from Producer\n"
              << "===========================\n";

    return 0;
}
