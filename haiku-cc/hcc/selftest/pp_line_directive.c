// EXPECT-EXIT: 0
int main(void) {
#line 500 "foo"
    if (__LINE__ != 500) return 1;
    if (__FILE__[0] != 'f') return 2;
#line 800 "bar"
    if (__LINE__ != 800) return 3;
#line 7
    if (__LINE__ != 7) return 4;
    return 0;
}
