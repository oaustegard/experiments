// EXPECT-EXIT: 0
// Float constants that overflow float (but not double) become infinity
// instead of crashing the compiler; in-range values are kept.
static float big = 3.5E+38f;
static float nbig = -3.5E+38f;
int main(void) {
    float local = 3.5E+38f;
    if (big < 1e38f) return 1;
    if (nbig > -1e38f) return 2;
    if (!(local > 1e38f)) return 3;
    if (!(-local < -1e38f)) return 4;
    if (1.9E+308 > 1e308 == 0) return 5;
    return 0;
}
