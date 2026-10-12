// EXPECT-EXIT: 0
// Vector parameters are not decayed to pointers; vector values are passed,
// returned and compound-assigned element-wise (scalar operand broadcast).
typedef unsigned char v8qi __attribute__((vector_size(8)));
typedef signed char v16 __attribute__((vector_size(16)));
static v8qi mul8(v8qi x, v8qi y) { return x * y; }
static v16 rem16(v16 c) { c %= (signed char) -19; return c; }
int main(void) {
    v8qi a = { 1, 2, 3, 4, 5, 6, 7, 8 };
    v8qi b = { 3, 3, 3, 3, 3, 3, 3, 3 };
    v8qi r = mul8(a, b);
    if (r[0] != 3 || r[7] != 24) return 1;
    v16 x = rem16((v16) { 31 });
    if (x[0] != 12) return 2;
    return 0;
}
