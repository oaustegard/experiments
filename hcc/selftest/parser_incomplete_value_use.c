// EXPECT-ERROR: [parse]
// An lvalue of incomplete struct type may not be read as a value (C11 6.3.2.1p2).
struct s;
extern struct s vs;
int main(void) { vs; return 0; }
