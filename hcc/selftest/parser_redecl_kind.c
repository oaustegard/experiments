// EXPECT-ERROR: redeclared
// Block-scope object repeats a parameter of the same function body (one scope).
int f(int a)
{
    int a[1];
    return a[0];
}
