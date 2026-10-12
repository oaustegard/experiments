// EXPECT-EXIT: 0
// A K&R definition whose float parameter matches a prior prototype `float f(float)`
// is accepted: the parameter is checked by knr_params, not the plain compat rule.
float f(float);
float f(a)
     float a;
{
  return a * 3;
}
int main(void) { return f(1.0f) != 3.0f; }
