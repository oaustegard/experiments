// EXPECT-EXIT: 0
// __builtin_nansf keeps its signaling bit pattern; fabs(float) clears only the sign.
extern void abort(void);
int main(void) {
    float s = __builtin_nansf("");
    unsigned int u;
    __builtin_memcpy(&u, &s, 4);
    if (u != 0x7fa00000u) abort();
    float a = __builtin_fabs(s);
    __builtin_memcpy(&u, &a, 4);
    if (u != 0x7fa00000u) abort();
    return 0;
}
