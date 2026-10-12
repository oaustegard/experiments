// EXPECT-EXIT: 0
// _Static_assert is accepted as the init clause of a for statement.
int main(void) {
    int i = 0;
    for (_Static_assert(1, "ok"); i < 3; i++)
        ;
    return i == 3 ? 0 : 1;
}
