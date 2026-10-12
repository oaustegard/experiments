// EXPECT-EXIT: 0
// Equal bytes but different element types (int vs unsigned) must not share a string object.
int main(void) { int a = L"\xffffffff"[0] >> 31; int b = U"\xffffffff"[0] >> 31; return (a == -1 && b == 1) ? 0 : 1; }
