// EXPECT-EXIT: 0
extern void abort(void);
char g[] = {"foo"};
char *p = (char []){"bar"};
int main(void) {
  char l[] = {"foo"};
  if (g[1] != 'o' || l[2] != 'o' || sizeof(l) != 4 || p[0] != 'b' || p[3] != 0)
    abort();
  return 0;
}
