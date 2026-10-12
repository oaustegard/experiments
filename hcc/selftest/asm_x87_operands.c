// EXPECT-EXIT: 0
// Extended asm with x87 operands: "=t" output, "0"/"u" inputs (st(0)/st(1)).
extern void abort(void);
int main(void) {
    long double r, x = 4.0L, y = 9.0L;
    asm("fsqrt" : "=t"(r) : "0"(x));
    if (r != 2.0L) abort();
    long double s;
    asm("fpatan" : "=t"(s) : "0"(1.0L), "u"(0.0L) : "st(1)");
    if (s != 0.0L) abort();
    long double q;
    asm("fsqrt" : "=t"(q) : "0"(y));
    if (q != 3.0L) abort();
    return 0;
}
