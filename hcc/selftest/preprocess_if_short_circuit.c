// EXPECT-EXIT: 0
#if 0 != (0 && (0/0))
#error and
#endif
#if 1 != (-1 || (0/0))
#error or
#endif
#if 3 != (-1 ? 3 : (0/0))
#error cond
#endif
int main(void) { return 0; }
