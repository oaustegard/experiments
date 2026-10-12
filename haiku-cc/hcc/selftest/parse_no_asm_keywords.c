// EXPECT-FLAGS: -fno-asm
// EXPECT-EXIT: 0
// With -fno-asm, asm/typeof are ordinary identifiers; inline stays a C99 keyword.
int asm;
int typeof;
int main(void) {
    asm = 1;
    typeof = 2;
    return asm + typeof - 3;
}
