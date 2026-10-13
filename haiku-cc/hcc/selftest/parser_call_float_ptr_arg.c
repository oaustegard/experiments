// EXPECT-ERROR: [parse]
// A floating argument is never converted to a pointer parameter (C11 6.5.16.1p1).
void h(int *p);
void k(double d) { h(d); }
int main(void) { return 0; }
