// EXPECT-ERROR: [parse]
void f(void) {
  goto a;
  ({ a: 0; });
}
