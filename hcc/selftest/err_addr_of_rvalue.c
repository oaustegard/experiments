// EXPECT-ERROR: lvalue required
// Unary '&' on a non-lvalue member access (struct returned by a comma expression).
struct s { char c[1]; };
struct s b;
int d;
int main(void) {
    char *p = &((d, b).c);
    return 0;
}
