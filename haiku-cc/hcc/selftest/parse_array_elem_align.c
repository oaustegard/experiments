// EXPECT-ERROR: size of array element is not a multiple of its alignment
typedef struct Bar { char c[129]; } Bar __attribute__((__aligned__(128)));
struct Foo { Bar bar[4]; };
int main(void) { return 0; }
