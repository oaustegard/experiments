// EXPECT-EXIT: 0
// GNU nested function definition inside a block (no outer-local access).
extern void abort(void);
int main(void) {
    int r;
    int twice(int x) {
        return x * 2;
    }
    r = twice(21);
    if (r != 42)
        abort();
    return 0;
}
