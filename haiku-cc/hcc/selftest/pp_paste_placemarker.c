// EXPECT-EXIT: 0
// ## on object-like macros pastes; chains with empty operands keep placemarkers;
// stringizing a pasted digraph keeps its spelling ("%:").
#define fifty int fif ## ty
#define glue3(x, y, z) x ## y ## z
#define str(x) #x
#define xstr(x) str(x)
#define glue(x, y) x ## y
fifty = 50;
int glue3(, , a) = 3;
int glue3(b, , ) = 4;
int main(void) {
  const char *s = xstr(glue(%, :)glue(%, :));
  return a + b - 7 + (s[0] == '%' && s[1] == ':' && s[2] == '%' ? 0 : 1);
}
