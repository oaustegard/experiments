// EXPECT-EXIT: 0
// GNU nested function definition returning a struct-with-VLA typedef (not a declaration).
int main(void) {
    int n = 3;
    typedef struct { char v[n]; } B;
    B __attribute__((noinline)) f(void) { B b; b.v[0] = 7; return b; }
    (void)f();
    return 0;
}
