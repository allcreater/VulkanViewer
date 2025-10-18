module;
#include <cassert>

export module utils.core;

import std;

export {

    template <std::ranges::bidirectional_range Range, typename Pr = std::equal_to<void>>
    auto intersectsFilter(Range && anotherRange, Pr&& predicate = {}) {
        assert(std::ranges::size(anotherRange) < 100); // for big cases better to do std::set_intersection

        return std::views::filter([anotherRange = std::forward<Range>(anotherRange), &predicate](const auto& x) {
            // return std::ranges::any_of(std::forward<Range>(anotherRange), [&x, &predicate](const auto& y) {
            //	return predicate(x, y);
            // });
            return std::ranges::any_of(std::forward<Range>(anotherRange), std::bind_front(predicate, x));
        });
    }

    template <typename T>
    requires std::is_trivially_copyable_v<T>
    std::vector<T> readFile(std::filesystem::path path) {
        std::basic_ifstream<char> stream{path, std::ios::binary | std::ios::in | std::ios::ate};
        stream.exceptions(std::ifstream::failbit);

        const auto dataLength = [&stream]() {
            auto pos = stream.tellg();
            stream.seekg(std::ios::beg);
            return static_cast<size_t>(pos);
        }();

        std::vector<T> data(dataLength / sizeof(T));
        std::copy(std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}, reinterpret_cast<char*>(data.data()));
        return data;
    }

    class NotMovableOrCopyable {
    public:
        NotMovableOrCopyable() = default;

        NotMovableOrCopyable(const NotMovableOrCopyable& _)            = delete;
        NotMovableOrCopyable(NotMovableOrCopyable&& _)                 = delete;
        NotMovableOrCopyable& operator=(const NotMovableOrCopyable& _) = delete;
        NotMovableOrCopyable& operator=(NotMovableOrCopyable&& _)      = delete;
    };

    class MoveOnly {
    public:
        MoveOnly() = default;

        MoveOnly(const MoveOnly& _)            = default;
        MoveOnly(MoveOnly&& _)                 = default;
        MoveOnly& operator=(const MoveOnly& _) = delete;
        MoveOnly& operator=(MoveOnly&& _)      = delete;
    };

    template <typename T>
    using OwningPtr = T*;
}
