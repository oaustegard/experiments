// EXPECT-EXIT: 53
#include <float.h>
#include <math.h>
/* DBL_MAX_10_EXP (308) + 1 (HUGE_VAL > 1e300) = 309, low byte 53 */
int main(void) {
    double h = HUGE_VAL;
    return DBL_MAX_10_EXP + (h > 1e300);
}
