#include "ds4_p2p.hpp"

#include <algorithm>
#include <sstream>
#include <utility>

namespace strata::ds4::p2p {
namespace {

std::string runtime_error(const char* operation, Error status, const Runtime& runtime) {
    std::ostringstream out;
    out << operation << " failed with runtime error " << status << " (" << runtime.error_string(status) << ")";
    return out.str();
}

void append_error(std::string& error, const std::string& next) {
    if (!error.empty()) error += "; ";
    error += next;
}

}  // namespace

PeerAccess::PeerAccess(Runtime& runtime, const std::vector<int>& devices)
    : runtime_(&runtime), devices_(devices) {}

PeerAccess::~PeerAccess() {
    std::string ignored;
    disable_all(ignored);
}

int PeerAccess::find_link(int device, int peer_device) const {
    for (std::size_t i = 0; i < links_.size(); ++i) {
        if (links_[i].pair.device == device && links_[i].pair.peer == peer_device) return static_cast<int>(i);
    }
    return -1;
}

bool PeerAccess::is_enabled(int device, int peer_device) const {
    const int index = find_link(device, peer_device);
    return index >= 0 && links_[static_cast<std::size_t>(index)].enabled;
}

bool PeerAccess::validate_devices(std::string& error) const {
    if (devices_.size() < 2) {
        error = "P2P requires an explicit list of at least two devices";
        return false;
    }
    for (std::size_t i = 0; i < devices_.size(); ++i) {
        if (devices_[i] < 0) {
            error = "P2P device index " + std::to_string(devices_[i]) + " is negative";
            return false;
        }
        for (std::size_t j = i + 1; j < devices_.size(); ++j) {
            if (devices_[i] == devices_[j]) {
                error = "P2P device index " + std::to_string(devices_[i]) + " occurs more than once";
                return false;
            }
        }
    }
    return true;
}

bool PeerAccess::restore_device(int device, std::string& error) const {
    const Error status = runtime_->set_device(device);
    if (status == 0) return true;
    append_error(error, runtime_error("set_device(restore)", status, *runtime_));
    return false;
}

bool PeerAccess::disable_owned(std::string& error) {
    bool ok = true;
    for (std::size_t n = owned_order_.size(); n > 0;) {
        const std::size_t order_index = --n;
        const std::size_t link_index = owned_order_[order_index];
        Link& link = links_[link_index];
        if (!link.owned) {
            owned_order_.erase(owned_order_.begin() + static_cast<std::ptrdiff_t>(order_index));
            continue;
        }
        const Error set_status = runtime_->set_device(link.pair.device);
        if (set_status != 0) {
            std::ostringstream operation;
            operation << "set_device(" << link.pair.device << ") before disable_peer_access(peer=" << link.pair.peer << ")";
            append_error(error, runtime_error(operation.str().c_str(), set_status, *runtime_));
            ok = false;
            continue;
        }
        const Error status = runtime_->disable_peer_access(link.pair.peer);
        if (status != 0) {
            std::ostringstream operation;
            operation << "disable_peer_access(device=" << link.pair.device << ",peer=" << link.pair.peer << ")";
            append_error(error, runtime_error(operation.str().c_str(), status, *runtime_));
            ok = false;
            continue;
        }
        link.owned = false;
        link.enabled = false;
        owned_order_.erase(owned_order_.begin() + static_cast<std::ptrdiff_t>(order_index));
    }
    return ok;
}

bool PeerAccess::enable_all(std::string& error) {
    error.clear();
    if (ready_) {
        error = "P2P enable_all called while the device set is already active";
        return false;
    }
    if (!owned_order_.empty() || std::any_of(links_.begin(), links_.end(), [](const Link& link) { return link.enabled; })) {
        error = "P2P enable_all refused: an earlier teardown is incomplete";
        return false;
    }
    if (!validate_devices(error)) return false;

    links_.clear();
    owned_order_.clear();
    for (int device : devices_) {
        for (int peer : devices_) {
            if (device != peer) links_.push_back(Link{DirectedPair{device, peer}, false, false});
        }
    }

    int original_device = -1;
    const Error get_status = runtime_->get_device(&original_device);
    if (get_status != 0) {
        error = runtime_error("get_device", get_status, *runtime_);
        links_.clear();
        return false;
    }

    bool ok = true;
    for (std::size_t i = 0; i < links_.size(); ++i) {
        Link& link = links_[i];
        const Error set_status = runtime_->set_device(link.pair.device);
        if (set_status != 0) {
            std::ostringstream operation;
            operation << "set_device(" << link.pair.device << ") before peer setup";
            error = runtime_error(operation.str().c_str(), set_status, *runtime_);
            ok = false;
            break;
        }

        int can_access = 0;
        const Error can_status = runtime_->can_access_peer(&can_access, link.pair.device, link.pair.peer);
        if (can_status != 0) {
            std::ostringstream operation;
            operation << "can_access_peer(device=" << link.pair.device << ",peer=" << link.pair.peer << ")";
            error = runtime_error(operation.str().c_str(), can_status, *runtime_);
            ok = false;
            break;
        }
        if (can_access != 1) {
            error = "P2P access unavailable for directed pair device=" + std::to_string(link.pair.device) +
                    ",peer=" + std::to_string(link.pair.peer) + " (can_access=" + std::to_string(can_access) + ")";
            ok = false;
            break;
        }

        const Error enable_status = runtime_->enable_peer_access(link.pair.peer, 0);
        if (enable_status == 0) {
            link.enabled = true;
            link.owned = true;
            owned_order_.push_back(i);
        } else if (enable_status == runtime_->peer_access_already_enabled()) {
            link.enabled = true;
            link.owned = false;
        } else {
            std::ostringstream operation;
            operation << "enable_peer_access(device=" << link.pair.device << ",peer=" << link.pair.peer << ")";
            error = runtime_error(operation.str().c_str(), enable_status, *runtime_);
            ok = false;
            break;
        }
    }

    if (ok) {
        ready_ = true;
    } else {
        std::string cleanup_error;
        disable_owned(cleanup_error);
        for (Link& link : links_) link.enabled = false;
        append_error(error, cleanup_error);
    }
    if (!restore_device(original_device, error)) ok = false;
    return ok;
}

bool PeerAccess::disable_all(std::string& error) {
    error.clear();
    if (!ready_ && owned_order_.empty()) return true;

    bool ok = true;
    if (!owned_order_.empty()) {
        int original_device = -1;
        const Error get_status = runtime_->get_device(&original_device);
        if (get_status != 0) {
            error = runtime_error("get_device before peer teardown", get_status, *runtime_);
            return false;
        }
        ok = disable_owned(error);
        if (!restore_device(original_device, error)) ok = false;
    }
    for (Link& link : links_) {
        if (!link.owned) link.enabled = false;
    }
    ready_ = false;
    return ok;
}

bool PeerAccess::copy_peer_async(void* dst, int dst_device, const void* src, int src_device,
                                 std::size_t bytes, Stream stream, std::size_t& bytes_enqueued,
                                 std::string& error) {
    bytes_enqueued = 0;
    error.clear();
    if (!ready_) {
        error = "P2P copy refused: directed peer access is not active";
        return false;
    }
    if (dst_device == src_device) {
        error = "P2P copy requires different source and destination devices";
        return false;
    }
    if (bytes != 0 && (dst == nullptr || src == nullptr)) {
        error = "P2P copy received a null pointer with a nonzero byte count";
        return false;
    }
    if (!is_enabled(dst_device, src_device)) {
        error = "P2P copy refused: destination->source link is not enabled (dst=" + std::to_string(dst_device) +
                ",src=" + std::to_string(src_device) + "); no host fallback is attempted";
        return false;
    }

    int original_device = -1;
    const Error get_status = runtime_->get_device(&original_device);
    if (get_status != 0) {
        error = runtime_error("get_device before peer copy", get_status, *runtime_);
        return false;
    }
    const Error set_status = runtime_->set_device(dst_device);
    if (set_status != 0) {
        error = runtime_error("set_device(destination) before peer copy", set_status, *runtime_);
        restore_device(original_device, error);
        return false;
    }

    const Error copy_status = runtime_->memcpy_peer_async(dst, dst_device, src, src_device, bytes, stream);
    if (copy_status != 0) {
        error = runtime_error("memcpy_peer_async", copy_status, *runtime_);
        restore_device(original_device, error);
        return false;
    }
    bytes_enqueued = bytes;
    return restore_device(original_device, error);
}

}  // namespace strata::ds4::p2p
