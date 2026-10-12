// EXPECT-EXIT: 0
// transparent_union: a call argument matching a member is passed through the union.
typedef union { int *ip; float *fp; } U __attribute__((__transparent_union__));
int take(U u) { return u.ip != 0 ? 1 : 2; }
int main(void) {
    int i = 0;
    float f = 0;
    return take(&i) + take(&f) - 2;
}
