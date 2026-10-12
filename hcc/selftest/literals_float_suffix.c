// EXPECT-EXIT: 0
// float literal suffixes w/W (__float80) and q/Q (__float128); __float80 type
__float80 a = 123.456789W;
__float80 b = 2.5w;
__float128 c = 1.0q;
int main(void) { return (a > 123.0 && b == 2.5 && c == 1.0) ? 0 : 1; }
