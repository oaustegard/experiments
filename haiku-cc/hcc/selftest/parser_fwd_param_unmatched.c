// EXPECT-ERROR: [parse] parameter 'a' has just a forward declaration
int h(int a; int b);
int main(void) { return 0; }
