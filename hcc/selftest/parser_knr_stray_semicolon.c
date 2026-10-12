// EXPECT-ERROR: [parse] expected declaration specifiers
// A stray ';' in an old-style (K&R) parameter declaration list is invalid.
void f(a)
     int a;;
{
}
