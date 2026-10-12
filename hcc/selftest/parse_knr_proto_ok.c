// EXPECT-EXIT: 0
// An old-style definition whose declared parameter types match the prior
// prototype (float stays float; not promoted) is accepted.
float g(float a);
float g(a) float a; { return a; }
int main(void) { return g(0.0f) != 0.0f; }
