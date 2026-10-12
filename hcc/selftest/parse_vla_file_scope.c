// EXPECT-ERROR: variably modified
// A non-constant array bound at file scope is invalid (C11 6.7.6.2p2).
int a;
int b0[a];
int main(void) { return 0; }
