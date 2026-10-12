// EXPECT-ERROR: [parse] braced-group within expression allowed only inside a function
// A GNU statement expression may not appear in the array size of a parameter
// declaration (prototype or K&R parameter list), even inside a function body.
int main(void) {
    void fun(a) int a[({int b = 2; b;})]; { }
    return 0;
}
