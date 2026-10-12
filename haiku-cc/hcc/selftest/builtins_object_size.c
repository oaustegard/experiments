// EXPECT-EXIT: 0
struct s { int i; char c[]; } s = { 1, "01234" };
struct t { int a; int b[4]; };
extern void abort(void);
int main(void) {
    struct t x;
    char buf[10];
    if (__builtin_object_size(&x.b, 0) != sizeof(x) - 4) return 1;
    if (__builtin_object_size(buf, 1) != 10) return 2;
    if (__builtin_object_size(&s.c, 0) != 6) return 3;
    if (__builtin_object_size(buf + 2, 1) != (unsigned long)-1) return 4;
    return 0;
}
