// EXPECT-ERROR: redeclared as different kind
typedef int a;
int a;
int main(void) { return 0; }
