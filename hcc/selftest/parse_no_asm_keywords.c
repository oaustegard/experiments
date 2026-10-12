// EXPECT-FLAGS: -fno-asm
// EXPECT-EXIT: 0
// With -fno-asm, plain asm/typeof/inline are ordinary identifiers.
int asm;
int typeof;
int inline;
int main(void) {
    asm = 1;
    typeof = 2;
    inline = 3;
    return asm + typeof + inline - 6;
}
