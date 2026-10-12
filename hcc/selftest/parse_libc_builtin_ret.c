// EXPECT-EXIT: 0
// Undeclared library builtins keep their real return type (not implicit int).
int main(void)
{
  int a[8];
  char *b = __builtin_memset(a, 0, sizeof a);
  a[0] = 1;
  if (b != (char *) a)
    __builtin_abort();
  if (memcpy(a, a, 0) == 0)
    return 1;
  return 0;
}
