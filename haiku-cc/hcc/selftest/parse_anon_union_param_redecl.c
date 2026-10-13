// EXPECT-EXIT: 0
/* An untagged union declared in a parameter list is an old GCC extension: a
   later prototype with an int parameter is accepted (with a warning). */
int f(union { int a; });
int f(int a);
int main(void) { return 0; }
