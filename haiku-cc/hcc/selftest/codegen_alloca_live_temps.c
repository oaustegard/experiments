// EXPECT-EXIT: 0
// alloca while the stack machine holds temporaries (store address pushed).
int main(void) {
    char *p;
    long n = 0;
    *(p = (char *)__builtin_alloca(16)) = 7;
    p[15] = 9;
    return !(p[0] == 7 && p[15] == 9) + (n != 0);
}
