// EXPECT-ERROR: [parse] '[*]' not allowed in other than function prototype scope
void f(int x[*]) { }
int main(void) {
    int a[*];
    return 0;
}
