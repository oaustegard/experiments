// EXPECT-EXIT: 3
// C11 Annex D.1 characters (superscript/vulgar fractions) may continue an identifier.
int main(void) { int a¾ = 1; int ab¾0 = 2; return a¾ + ab¾0 * 0 + 2; }
