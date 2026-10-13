// EXPECT-ERROR: too few arguments
// __atomic_exchange_n takes exactly 3 arguments.
int i;
int main(void) {
    __atomic_exchange_n(&i, 1);
    return 0;
}
