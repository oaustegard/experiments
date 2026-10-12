// EXPECT-EXIT: 0
// A 24-bit `long` bit-field promotes to int, so it compares equal to an
// unsigned int literal with the same bits after sign extension.
extern void abort(void);
typedef struct { long int a : 24; long int b : 24; } st;
int main(void) {
    st s;
    s.a = 0xFEFEFEFE;
    if (s.a != 0xFFFEFEFE) abort();
    return 0;
}
