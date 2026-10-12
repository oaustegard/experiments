// EXPECT-ERROR: [parse] thread-local declaration of 'i' follows non-thread-local
// A file-scope declaration must agree with earlier ones on thread-local-ness.
int i;
__thread int i;
int main(void) { return 0; }
