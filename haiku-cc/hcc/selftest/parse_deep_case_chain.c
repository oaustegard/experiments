// EXPECT-EXIT: 7
// A switch with ~10000 directly consecutive case labels: the label chain nests
// deeply in the AST and must neither exhaust recursion nor be scanned quadratically.
#define L1(x) x##0: x##1: x##2: x##3: x##4: x##5: x##6: x##7: x##8: x##9:
#define L2(x) L1(x##0) L1(x##1) L1(x##2) L1(x##3) L1(x##4) L1(x##5) L1(x##6) L1(x##7) L1(x##8) L1(x##9)
#define L3(x) L2(x##0) L2(x##1) L2(x##2) L2(x##3) L2(x##4) L2(x##5) L2(x##6) L2(x##7) L2(x##8) L2(x##9)
#define L4(x) L3(x##0) L3(x##1) L3(x##2) L3(x##3) L3(x##4) L3(x##5) L3(x##6) L3(x##7) L3(x##8) L3(x##9)

int main(void) {
    long i = 19999;
    int r = 0;
    switch (i) {
        L4(case 1)
            r = 7;
            break;
    }
    return r;
}
