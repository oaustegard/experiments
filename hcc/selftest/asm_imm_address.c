// EXPECT-EXIT: 0
// Asm "i" operands accept address constants (&global + n) and pointer
// differences within one array parameter (folded to a byte offset).
static char var;
static char arr[16][16];

void f(char a[16][16][16]) {
    asm volatile("" : : "i"(&a[1][0][0] - &a[0][0][0]));
}

int main(void) {
    asm volatile("" : : "i"(&var + 1));
    asm volatile("" : : "i"(&arr[2][3] - &arr[0][0]));
    f(0);
    return 0;
}
