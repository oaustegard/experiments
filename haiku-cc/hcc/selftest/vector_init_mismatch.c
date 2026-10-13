// EXPECT-ERROR: incompatible types
// vector initializers must match element type and subpart count (no implicit vector conversion)
typedef int v4si __attribute__((vector_size(16)));
typedef short v8hi __attribute__((vector_size(16)));
struct s { v8hi x; v4si y; };
void foo(v8hi h) {
    struct s x2 = { h, h };
}
int main(void) { return 0; }
