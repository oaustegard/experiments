// EXPECT-EXIT: 0
void foo(void);
void (*fp)(void);
char x[sizeof (1, foo) == sizeof (fp) ? 1 : -1];
int main(void) { return sizeof x - 1; }
