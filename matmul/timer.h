#pragma once
#include <cstdint>

using namespace std;

class timer
{
    uint64_t t0;

public:
    timer();
    double elapsed() const;
    double reset();
};