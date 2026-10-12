// EXPECT-ERROR: [parse]
// Attributes written after the declarator of a function definition are invalid.
void foo(void) __attribute__((__visibility__("default"))) { }
int main(void) { return 0; }
