// EXPECT-ERROR: flexible array member in union
// A union may not contain a flexible array member, even through a typedef.
typedef char T[];
union u { int a; T b; };
int main(void) { return 0; }
