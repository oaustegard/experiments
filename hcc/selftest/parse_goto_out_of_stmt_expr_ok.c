// EXPECT-EXIT: 7
int f(void) {
  int r = ({ goto out; 1; });
  return r;
out:
  return 7;
}
int main(void) { return f(); }
