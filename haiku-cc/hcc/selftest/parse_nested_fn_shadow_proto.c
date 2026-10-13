// EXPECT-EXIT: 5
// A nested function definition hides an outer-scope prototype of another type.
void f(long a, char b);
int main(void) {
    int r = 0;
    void f(int a, int b) { r = a + b; }
    f(2, 3);
    return r;
}
