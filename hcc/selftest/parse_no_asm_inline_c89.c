// EXPECT-FLAGS: -std=gnu89 -fno-asm
// EXPECT-EXIT: 0
// Under -fno-asm in gnu89, inline is an ordinary identifier (no empty-declaration error).
int inline;
int main(void) { inline = 0; return inline; }
