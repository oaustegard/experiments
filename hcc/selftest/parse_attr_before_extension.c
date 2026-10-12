// EXPECT-EXIT: 7
// EXPECT-FLAGS: -std=gnu2x
// C23 attribute lists may follow __extension__ and precede a declaration.
__extension__ [[]] int g = 4;
int main(void) { __extension__ [[]] int b = 3; return g + b; }
