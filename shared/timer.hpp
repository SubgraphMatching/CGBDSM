#pragma once
#include <sys/time.h>

class Timer {
public:
    Timer() = default;
    void Start() { gettimeofday(&start_time_, nullptr); }
    // Returns elapsed milliseconds since Start(), also stops the timer.
    float Finish() {
        struct timeval end;
        gettimeofday(&end, nullptr);
        return (end.tv_sec - start_time_.tv_sec) * 1000.0f
             + (end.tv_usec - start_time_.tv_usec) / 1000.0f;
    }
private:
    struct timeval start_time_{};
};
