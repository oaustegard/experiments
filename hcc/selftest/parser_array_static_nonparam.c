// EXPECT-ERROR: static or type qualifiers in non-parameter array declarator
// C99 6.7.5.2p1: `static` in an array declarator is only valid for a parameter.
int main(void) { int a[static 2]; return a[0]; }
