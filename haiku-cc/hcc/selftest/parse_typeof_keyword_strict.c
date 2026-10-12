// EXPECT-EXIT: 5
// EXPECT-FLAGS: -std=c11
// In strict ISO mode `typeof` is an ordinary identifier, not a keyword.
int typeof = 2;
int main(void) { return typeof + 3; }
