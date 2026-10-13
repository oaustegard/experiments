// EXPECT-EXIT: 1
// A block-scope extern object does not clash with a later file-scope typedef of the same name.
int f (void)
{
  extern int t;
  return 0;
}

typedef float t;
t v = 4.5f;

int main (void)
{
  return v > 4.0f;
}
