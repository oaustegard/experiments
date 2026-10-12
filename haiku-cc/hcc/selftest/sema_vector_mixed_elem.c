// EXPECT-ERROR: invalid operands to binary
typedef float v4sf __attribute__((vector_size(16)));
typedef int v4si __attribute__((vector_size(16)));
v4sf b;
v4si d;
int main(void) {
    b + d;
    return 0;
}
