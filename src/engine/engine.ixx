module;
#include <memory>

export module engine;

export class IEngine {
public:
    virtual void Render() = 0;

    virtual ~IEngine() = default;
};

export std::unique_ptr<IEngine> MakeEngine(void* window);
