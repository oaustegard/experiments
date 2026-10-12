// EXPECT-ERROR: non-nested function with variably modified type
// A block-scope function declaration may not return a variably modified type.
int main(void) {
    int n = 3;
    typedef int A[n];
    A *g(void);
    return 0;
}
