// EXPECT-ERROR: incompatible types
// A function returning a struct cannot return a scalar value.
typedef struct { int x, y; } point_t;
point_t f(void) { return 0; }
int main(void) { return 0; }
