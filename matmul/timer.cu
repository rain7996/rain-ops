#include <chrono>

#include "timer.h"

using namespace std;

namespace
{
    uint64_t now()
    {
        using clock = chrono::steady_clock;
        return chrono::duration_cast<chrono::nanoseconds>(clock::now().time_since_epoch()).count();
    }
}

timer::timer()
{
    t0 = now();
}

double timer::elapsed() const
{
    uint64_t t1 = now();
    return 1.e-9 * (t1 - t0);
}

double timer::reset()
{
    uint64_t t1 = now();
    double ans = 1.e-9 * (t1 - t0);
    t0 = t1;
    return ans;
}