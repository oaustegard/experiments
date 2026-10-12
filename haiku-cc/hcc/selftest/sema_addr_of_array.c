// EXPECT-EXIT: 0
// &array yields a pointer to the whole array: pointer arithmetic steps by
// the array size, and *(&a[0][0]) is the first element.
extern void abort(void);
int main(void) {
    char a[2][4] = {"ab", "cd"};
    char *p = &a[0][0];
    if (*(&a[0][0] + 4) != 'c') abort();
    if (*(&a[0][0]) != 'a') abort();
    if ((char *)&a[1] - (char *)&a[0] != 4) abort();
    return 0;
}
