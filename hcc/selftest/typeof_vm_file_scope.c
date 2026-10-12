// EXPECT-EXIT: 0
// A variably modified type named inside a typeof operand at file scope is not an object declaration.
int n = 3;
int s = sizeof (typeof (int (*)[n]));
int main(void) {
    return s == (int)sizeof(void *) ? 0 : 1;
}
