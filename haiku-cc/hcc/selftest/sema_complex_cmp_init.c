// EXPECT-EXIT: 0
// A complex ==/!= between a complex and a real constant folds to an integer constant
// usable as a file-scope static initializer.
int f = (_Complex float)(0.5) == 0.5;
int g = (_Complex double)(2.0) != 2.0;
int main(void) {
    return (f == 1 && g == 0) ? 0 : 1;
}
