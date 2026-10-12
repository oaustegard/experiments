// EXPECT-EXIT: 0
// C23 storage class specifiers in compound literals: `static` gives static
// storage (initialized once), `register` is accepted; sizeof accepts them too.
int *gp = &(static int){ 1 };
int main(void) {
    int i = 0;
    int *p;
step:
    p = &(static int){ 0 };
    if (*p != i)
        return 1;
    i++;
    *p = i;
    if (i < 3)
        goto step;
    if ((register int){ 3 } != 3)
        return 2;
    if (sizeof (static int){ 1 } != sizeof(int))
        return 3;
    return *gp != 1;
}
