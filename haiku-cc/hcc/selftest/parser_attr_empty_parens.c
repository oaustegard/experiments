// EXPECT-ERROR: [parse]
// C23 [[deprecated()]]: empty argument list must omit the parentheses.
[[deprecated()]] int a;
int main(void) { return 0; }
