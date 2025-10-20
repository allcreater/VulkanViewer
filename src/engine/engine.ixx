module;
export module engine;

import std;

export class IEngine {
public:
    virtual void Render() = 0;
    virtual void LoadModel(const std::filesystem::path& path) = 0;

    virtual ~IEngine() = default;
};

export std::unique_ptr<IEngine> MakeEngine(void* window);
