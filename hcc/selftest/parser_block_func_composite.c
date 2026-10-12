// EXPECT-EXIT: 0
// A block-scope function declaration with an incomplete array element in the
// return pointer takes the complete type of the prior file-scope declaration.
typedef int A10[10];
A10 *c5(void);
int main(void) {
    {
        extern int (*c5(void))[];
        if (sizeof(*c5()) != sizeof(A10)) return 1;
    }
    return sizeof(*c5()) == sizeof(A10) ? 0 : 2;
}
