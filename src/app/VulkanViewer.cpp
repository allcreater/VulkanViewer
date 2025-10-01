// VulkanViewer.cpp : Defines the entry point for the application.
//

#include <SDL3/SDL.h>
#include "SDL3/SDL_vulkan.h"

import std;
import engine;

class SdlApp {
    using SdlWindowDeleter = decltype([](SDL_Window* window) { SDL_DestroyWindow(window); });

public:
    void Run() {
        SDL_Init(SDL_INIT_VIDEO);
        std::unique_ptr<SDL_Window, SdlWindowDeleter> window{
            SDL_CreateWindow("window", 1024, 768, SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_VULKAN)};

        // SDL_Vulkan_CreateSurface(window.get(), )
        auto engine = MakeEngine(window.get());

        bool run = true;
        while (run) {
            for (SDL_Event event; SDL_PollEvent(&event) > 0;) {
                if (event.type == SDL_EVENT_QUIT)
                    run = false;
            }
            engine->Render();
            // SDL_Delay(1);
        }
    }

private:
};


int main(int argc, char** argv) {
    SdlApp app;
    app.Run();

    return 0;
}
