// EXPECT-EXIT: 0
// Vector_size inside a type name (compound literal / cast) must keep the vector type.
int main(void) {
    __attribute__((vector_size(16))) float f = (__attribute__((vector_size(16))) float){1.f, 2.f, 3.f, 4.f};
    __attribute__((vector_size(16))) float g;
    g = (__attribute__((vector_size(16))) float){2.f, 2.f, 2.f, 2.f} + f;
    if ((int) g[0] != 3) return 1;
    if ((int) g[3] != 6) return 2;
    return 0;
}
