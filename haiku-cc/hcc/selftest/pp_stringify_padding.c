// EXPECT-EXIT: 0
// #x keeps a space where an empty macro in the argument's expansion sat between
// two tokens (GCC padding rules); an empty expansion at the edges adds none.
extern int strcmp(const char *, const char *);
#define E
#define EF()
#define S(x) #x
#define V(x) S(x)
#define W(x) x E
#define Z() EF() E
int main(void)
{
  if (strcmp(V(|E|), "||")) return 1;
  if (strcmp(V(| E |), "| |")) return 2;
  if (strcmp(V(|W(|)), "||")) return 3;
  if (strcmp(V(|Z()|), "| |")) return 4;
  if (strcmp(V(| Z()|), "| |")) return 5;
  return 0;
}
