// EXPECT-ERROR: used union type value where scalar is required
union a { int x; };
int f(union a a) { if (a) return 1; return 0; }
