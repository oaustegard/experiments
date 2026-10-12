// EXPECT-ERROR: [parse]
// _Alignas in a compound literal's type name cannot reduce the type's alignment.
int *p = &(_Alignas(1) int){ 1 };
int main(void) { return 0; }
