// EXPECT-EXIT: 0
// C23 [[...]] attributes after a declarator name, before a pointer's qualifiers,
// and on struct members and parameters.
struct s { [[gnu::unused]] int m; };
void f [[]] (void);
int g(int [[]] x, [[]] int y) { return x + y; }
int *[[]] const p;
int main(void) {
    struct s v;
    v.m = g(1, 2) - 3;
    return v.m;
}
