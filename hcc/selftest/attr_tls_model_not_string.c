// EXPECT-ERROR: argument not a string
// __attribute__((tls_model(N))) with a non-string argument is a hard error.
__thread int a __attribute__((tls_model(1)));
int main(void) { return 0; }
