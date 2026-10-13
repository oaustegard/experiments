// EXPECT-ERROR: [parse]
// A file-scope tentative definition of a struct that is never completed is rejected
// (C11 6.9.2p2); a later completion in the same TU makes it valid.
struct S;
struct S a;
int main(void) { return 0; }
