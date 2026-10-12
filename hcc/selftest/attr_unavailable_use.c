// EXPECT-ERROR: [parse]      (compile must fail; stderr must contain text)
// EXPECT-ERROR: is unavailable
// __attribute__((unavailable)) on a typedef, a variable and a function: any use is an error.
typedef int T __attribute__((unavailable));
int v __attribute__((unavailable));
void f(void) __attribute__((__unavailable__));
int main(void) {
    f();
    return 0;
}
