// EXPECT-EXIT: 0
#define ATTR(list) __attribute__ (list)
int n;
int main(void) {
    if (__builtin_has_attribute(struct A { int i; }, aligned) != 0) return 1;
    if (__builtin_has_attribute(ATTR((aligned)) int, aligned) != 1) return 2;
    if (__builtin_has_attribute(ATTR((aligned)) int[n++], aligned) != 1) return 3;
    if (__builtin_has_attribute(n, aligned) != 0) return 4;
    if (n != 0) return 5;
    return 0;
}
