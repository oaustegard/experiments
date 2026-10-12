// EXPECT-EXIT: 0
// __builtin_modff16 stores the integral part as _Float16 through the pointer;
// __builtin_assoc_barrier is the identity on its operand.
extern void abort(void);
int main(void) {
    _Float16 ip = 0, fr;
    fr = __builtin_modff16(5.75f16, &ip);
    if (ip != 5.0f16 || fr != 0.75f16) abort();
    if (__builtin_assoc_barrier(ip + fr) - 5.75f16 != 0) abort();
    return 0;
}
