// EXPECT-ERROR: incompatible types
typedef char v8 __attribute__((vector_size(8)));
typedef short v4s __attribute__((vector_size(8)));
v8 y = (v4s){1,2,3,4};
int main(void) { return 0; }
