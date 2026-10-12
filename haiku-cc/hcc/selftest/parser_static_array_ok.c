// EXPECT-EXIT: 0
// `static` with a size expression, or with `*n`, is valid in a parameter.
void fa(int *n, char buf[static *n]);
void fe(int n, char buf[static n]);
void fs(char buf[static 4]);
int main(void) { return 0; }
