// EXPECT-EXIT: 0
// Alignment exactly at the object file maximum is accepted.
int foo __attribute__ ((aligned(1 << 28))) = 20;
int main(void) { return 0; }
