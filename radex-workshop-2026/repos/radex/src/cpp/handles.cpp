#include "radex/handles.hpp"

#include <sstream>
#include <string>

namespace radex::data {

auto IncomingHandle::key() const -> std::string {
    std::ostringstream key;
    key << name;
    return key.str();
}

auto IncomingHandle::metadata_key() const -> std::string {
    std::ostringstream meta;
    meta << "metadata::" << key();
    return meta.str();
}

auto OutgoingHandle::key() const -> std::string {
    std::ostringstream key;
    key << name;
    return key.str();
}

auto OutgoingHandle::metadata_key() const -> std::string {
    std::ostringstream meta;
    meta << "metadata::" << key();
    return meta.str();
}

} // namespace radex::data
