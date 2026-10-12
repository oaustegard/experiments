// EXPECT-EXIT: 0
// EXPECT-FLAGS: -std=c23
// C23: true/false are bool constants; in #if they evaluate to 1 / 0.
#if true != 1 || false != 0
#error "true/false in #if"
#endif
int main(void) { return true ? 0 : 1; }
