module;
#include <algorithm>
#include <memory>
#include <ranges>
#include <functional>

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

}