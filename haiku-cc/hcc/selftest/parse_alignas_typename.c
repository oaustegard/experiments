// EXPECT-ERROR: [parse] '_Alignas' specified for type name
// _Alignas is a declaration specifier; it may not appear in a type name (cast, sizeof).
int main(void) { return (int)sizeof(_Alignas(8) long); }
