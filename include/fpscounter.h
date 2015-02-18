#include <chrono>
#include <deque>

class fps_counter {
    std::deque<decltype(std::chrono::system_clock::now())> samples;
    unsigned window_size;

  public:
    fps_counter(unsigned _window_size = 20) : window_size(_window_size) {}

    void tick() {
        samples.emplace_back(std::chrono::system_clock::now());
        if (samples.size() > window_size) {
            samples.pop_front();
        }
    }

    float get() const {
        if (samples.size() < 2) {
            return 0.0;
        }
        auto d = samples.back() - samples.front();
        auto dms = std::chrono::duration_cast<std::chrono::microseconds>(d);
        double frame_d = (double)dms.count() / (samples.size() - 1);
        return 1000000 / frame_d;
    }
};
