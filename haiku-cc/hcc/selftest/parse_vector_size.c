// EXPECT-EXIT: 0
// vector_size typedefs: value init, copy-assign, subscript
typedef long vec2 __attribute__((vector_size(16)));
vec2 g = { 3, 4 };
vec2 mk(long a) { vec2 v = { a, a + 1 }; return v; }
int main(void) {
    vec2 a = g;
    vec2 b;
    b = a;
    vec2 c = mk(5);
    if (b[0] != 3 || b[1] != 4) return 1;
    if (c[0] != 5 || c[1] != 6) return 2;
    return 0;
}
