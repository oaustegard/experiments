// EXPECT-EXIT: 0
// __builtin_signbitf/signbitl are inline (no libc call), like signbit.
extern void abort(void);
int main(void) {
    if (!__builtin_signbitf(-0.0f) || __builtin_signbitf(1.0f)) abort();
    if (!__builtin_signbitl(-2.0L) || __builtin_signbitl(3.0L)) abort();
    return 0;
}
