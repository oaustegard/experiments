// EXPECT-EXIT: 0
struct s { int a[3]; int c[3]; int z; };
struct s g = { c: {1, 2, 3}, z: 7 };
int main(void) {
    struct s l = { z: 4, a: {5} };
    if (g.c[0] != 1 || g.c[2] != 3 || g.z != 7) return 1;
    if (l.a[0] != 5 || l.a[1] != 0 || l.z != 4) return 2;
    return 0;
}
