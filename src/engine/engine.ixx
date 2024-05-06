module;
#include <memory>

export module engine;

export class IEngine {
public:
	virtual ~IEngine() = default;
};

export std::unique_ptr<IEngine> MakeEngine(void* window);