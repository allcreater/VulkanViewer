module;
#include <algorithm>
#include <memory>
#include <ranges>
#include <functional>
#include <vector>
#include <fstream>
#include <filesystem>

#include <vulkan/vulkan_raii.hpp>

export module engine : utils;

export {

template <std::ranges::bidirectional_range Range, typename Pr = std::equal_to<void>>
auto intersectsFilter(Range&& anotherRange, Pr&& predicate = {}) {
	assert(std::ranges::size(anotherRange) < 100); // for big cases better to do std::set_intersection

	return std::views::filter([anotherRange = std::forward<Range>(anotherRange), &predicate](const auto& x) {
		//return std::ranges::any_of(std::forward<Range>(anotherRange), [&x, &predicate](const auto& y) {
		//	return predicate(x, y);
		//});
		return std::ranges::any_of(std::forward<Range>(anotherRange), std::bind_front(predicate, x));
		});
}

std::vector<char> readFile(std::filesystem::path path) {
	std::basic_ifstream<char> stream{ path, std::ios::binary | std::ios::in | std::ios::ate };
	stream.exceptions(std::ifstream::failbit);

	const auto dataLength = [&stream]() {
		auto pos = stream.tellg();
		stream.seekg(std::ios::beg);
		return static_cast<size_t>(pos);
		}();

	std::vector<char> data(dataLength);
	data.assign(std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{});
	return data;
}

}