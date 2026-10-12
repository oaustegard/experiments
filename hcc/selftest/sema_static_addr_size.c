// EXPECT-ERROR: not constant
// A file-scope char cannot be initialised with an address constant (gcc rejects it).
char c = (char)"abc";
int main(void) { return c; }
