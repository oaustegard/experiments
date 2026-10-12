// EXPECT-EXIT: 0
int g[4];
int calls;
int bump(void) { return ++calls; }
int main(void) {
    int *p = g;
    __builtin_prefetch(p, 0, 0);
    __builtin_prefetch(g, 1, 3);
    __builtin_prefetch(&g[1]);
    __builtin_prefetch(p + bump(), 0, 1);
    if (calls != 1) return 1;
    return 0;
}
