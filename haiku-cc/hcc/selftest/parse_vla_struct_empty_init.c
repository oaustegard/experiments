// EXPECT-EXIT: 0
// A struct with a run-time sized member may be initialized with empty braces (zero-filled).
int f(int a) {
    struct s { int x[a]; };
    struct s b = {};
    int sum = 0;
    for (int i = 0; i < a; i++)
        sum += b.x[i];
    return sum;
}
int main(void) {
    return f(5) + f(3);
}
