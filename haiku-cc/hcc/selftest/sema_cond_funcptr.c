// EXPECT-EXIT: 0
static int ret10(void) { return 10; }
int main(void) {
    int r = (1 ? ret10 : (void *)0)();
    int *p = 0;
    int x = 3;
    int *q = 0 ? (void *)0 : &x;
    *q = 4;
    return !(r == 10 && x == 4 && p == 0);
}
