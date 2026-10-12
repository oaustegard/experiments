// EXPECT-ERROR: [parse]
void f(int n) {
  goto x;
  int a[n];
x:
  ;
}
int main(void) { return 0; }
