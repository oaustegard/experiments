// EXPECT-EXIT: 0
// A `#pragma weak` placed before a tentative definition makes the symbol weak:
// it must go to .bss, not .comm (the assembler rejects weak + common).
#pragma weak wvar
int wvar;
int main(void) {
    return wvar;
}
