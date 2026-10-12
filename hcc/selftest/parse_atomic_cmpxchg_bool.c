// EXPECT-EXIT: 0
// __atomic_compare_exchange_n yields _Bool regardless of the operand type.
int main(void) {
    char c = 0, e = 0;
    _Generic(__atomic_compare_exchange_n(&c, &e, 1, 0, 0, 0),
             _Bool: (void)0, default: (void)(1 / 0));
    return 0;
}
