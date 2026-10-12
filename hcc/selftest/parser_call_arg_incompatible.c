// EXPECT-ERROR: [parse]
// A float argument passed to a pointer parameter is a constraint violation.
extern int takes_ptr (const char *p);
int f (float x)
{
  return takes_ptr (x);
}
