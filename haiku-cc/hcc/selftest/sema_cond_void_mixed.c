// EXPECT-EXIT: 0
// A void operand makes the conditional void even when the other operand is not void.
int g;
int main(void) {
    int c = 1;
    c ? (void)0 : g;
    1 ? -2 : (void)-1;
    c ?: (void)0;
    return 0;
}
