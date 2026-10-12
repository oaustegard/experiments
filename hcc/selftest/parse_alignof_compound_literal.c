// EXPECT-EXIT: 0
// _Alignof applied to a compound literal: (type){...} after _Alignof is an operand, not a type name.
int main(void) {
    unsigned long a = _Alignof (long) { 1 };
    return a == 8 ? 0 : 1;
}
