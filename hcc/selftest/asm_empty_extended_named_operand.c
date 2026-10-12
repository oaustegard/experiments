// EXPECT-ERROR: [codegen]
// Extended asm with no operands still substitutes its template: an unknown
// %[name] reference is an error.
void h(void) { asm("%[x]" : : ); }
int main(void) { return 0; }
