#include "radex/client.hpp"
#include "radex/constants.hpp"
#include "radex/handles.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace radex {
namespace detail {

BytesBuffer::BytesBuffer(BytesBuffer &&other) noexcept
    : data{std::move(other.data)}, length{other.length} {
    other.length = 0;
};

BytesBuffer &BytesBuffer::operator=(BytesBuffer &&other) noexcept {
    if (this != &other) {
        length = other.length;
        data = other.release();
    }
    return *this;
}

std::unique_ptr<std::uint8_t[]> BytesBuffer::release() noexcept {
    length = 0;
    return std::move(data);
}

MetaData MetaData::from_buffer(BytesBuffer buffer) {
    MetaData meta{std::move(buffer)};
    MetaInt expected_size =
        (meta.n_dims() + Index::END_OF_HEADER) * sizeof(MetaInt);
    if (meta.size() != expected_size) {
        throw radex::MetadataError("Malformed item metadata buffer received");
    }
    return meta;
}

MetaData MetaData::make_buffer(const radex::data::DType dtype,
                               const MetaInt *dims, MetaInt n_dims) {
    MetaInt n_bytes = (n_dims + Index::END_OF_HEADER) * sizeof(MetaInt);
    auto bytes = std::make_unique<std::uint8_t[]>(n_bytes);

    auto buf = static_cast<MetaInt *>(static_cast<void *>(bytes.get()));
    buf[Index::TYPE] = static_cast<MetaInt>(dtype);
    buf[Index::N_DIMS] = n_dims;
    std::memcpy(buf + Index::END_OF_HEADER, dims, n_dims * sizeof(*dims));

    return {BytesBuffer{std::move(bytes), n_bytes}};
}

} // namespace detail

detail::BytesBuffer IClient::wait_for_bytes(std::string_view key,
                                            std::chrono::milliseconds timeout) {
    const char *poll_interval_s = nullptr;
    const auto end_time = std::chrono::steady_clock::now() + timeout;
    const auto poll_interval = std::chrono::milliseconds(
        (poll_interval_s = std::getenv(RADEX_POLL_INTERVAL_VAR.c_str()))
            ? std::stoul(poll_interval_s)
            : RADEX_DEFAULT_POLL_INTERVAL_MILLISECONDS);

    while (!contains(key)) {
        const auto curr_time = std::chrono::steady_clock::now();
        const auto timeout_expired = curr_time > end_time;

        if (timeout_expired) {
            std::ostringstream msg;
            msg << "Failed to find key `" << key << "` before timeout";

            throw radex::TimeoutError(msg.str());
        }
        std::this_thread::sleep_for(poll_interval);
    }
    return get_bytes(key);
}

detail::ItemInfo IClient::get_item_info(
    std::function<detail::BytesBuffer(std::string_view)> fetch_bytes,
    const data::IncomingHandle &handle) {
    auto buf = fetch_bytes(handle.key());
    return {get_meta_data(handle), buf.release()};
}

std::unique_ptr<detail::ItemInfo>
IClient::get_item_info_ptr(const data::IncomingHandle &handle) {
    const auto fetch_bytes =
        std::bind(&IClient::get_bytes, this, std::placeholders::_1);
    return std::make_unique<detail::ItemInfo>(
        get_item_info(fetch_bytes, handle));
}

std::unique_ptr<detail::ItemInfo>
IClient::wait_for_item_info_ptr(const data::IncomingHandle &handle,
                                std::chrono::milliseconds timeout) {
    const auto fetch_bytes = std::bind(&IClient::wait_for_bytes, this,
                                       std::placeholders::_1, timeout);
    return std::make_unique<detail::ItemInfo>(
        get_item_info(fetch_bytes, handle));
}

void IClient::delete_item(const data::OutgoingHandle &handle) {
    delete_key(handle.key());
    if (contains(handle.metadata_key())) {
        delete_key(handle.metadata_key());
    }
}

detail::MetaData IClient::get_meta_data(const data::IncomingHandle &handle) {
    return detail::MetaData::from_buffer(get_bytes(handle.metadata_key()));
}

} // namespace radex
