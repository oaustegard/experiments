// EXPECT-EXIT: 0
// Deeply nested parentheses and a long chain of case labels must not
// exhaust the parser's recursion limit.
#define P1 ( ( ( ( ( ( ( ( ( (
#define P2 P1 P1 P1 P1 P1 P1 P1 P1 P1 P1
#define P3 P2 P2 P2 P2 P2 P2 P2 P2 P2 P2
#define P4 P3 P3 P3 P3 P3 P3 P3 P3 P3 P3
#define Q1 ) ) ) ) ) ) ) ) ) )
#define Q2 Q1 Q1 Q1 Q1 Q1 Q1 Q1 Q1 Q1 Q1
#define Q3 Q2 Q2 Q2 Q2 Q2 Q2 Q2 Q2 Q2 Q2
#define Q4 Q3 Q3 Q3 Q3 Q3 Q3 Q3 Q3 Q3 Q3
int v = P4 1 Q4;
#define C1 case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8: case 9: case 10:
#define C2 C1 C1 C1 C1 C1 C1 C1 C1 C1 C1
#define C3 C2 C2 C2 C2 C2 C2 C2 C2 C2 C2
int f(int i) {
    switch (i) {
    C3
        return 7;
    }
    return 0;
}
int main(void) { return v == 1 ? f(1) - 7 : 1; }
