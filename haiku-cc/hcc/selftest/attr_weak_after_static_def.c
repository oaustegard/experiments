// EXPECT-ERROR: [parse] weak declaration of 'foo' being applied to a already existing, static definition
static int foo (void) { return 0; }
int foo (void) __attribute__((weak));
int main(void) { return foo(); }
