// EXPECT-ERROR: [parse] '_Alignas' specifiers cannot reduce alignment
// _Alignas(N) may not lower the alignment below the type's natural alignment.
_Alignas(char) short int sc;
int main(void) { return 0; }
