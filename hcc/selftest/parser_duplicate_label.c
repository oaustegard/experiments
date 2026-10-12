// EXPECT-ERROR: [parse]
// Labels have function scope: the same label name twice in one function is an error.
void f(int i) {
 lbl:
  i++;
 lbl:
  i++;
}
