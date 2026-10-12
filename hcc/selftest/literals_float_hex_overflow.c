// EXPECT-EXIT: 0
// Hex float constants beyond double range become infinity (no crash);
// decimal float constants round directly to binary32 (no double rounding).
static const float big = 0x1.p+1026f128 > 1.0 ? 1.0f : 0.0f;
int main(void) {
    double inf = 0x1.p+5000;
    float f = 7.0064923216240853546186479164495806564013097093825788587853e-46f;
    unsigned int u;
    __builtin_memcpy(&u, &f, 4);
    if (inf != 1.0 / 0.0) return 1;
    if (big != 1.0f) return 2;
    if (u != 0u) return 3;
    return 0;
}
