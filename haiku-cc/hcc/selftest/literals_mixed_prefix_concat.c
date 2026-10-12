// EXPECT-ERROR: non-standard concatenation
// Adjacent string literals with distinct encoding prefixes (L and u8) are
// rejected, as GCC does.
const void *p = L"a" u8"b";
int main(void) { return 0; }
