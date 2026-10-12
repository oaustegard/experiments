// EXPECT-EXIT: 7
// GNU nested function with K&R-style parameter declarations inside a block.
int outer(void) {
    int twice(x)
        int x;
    {
        return x * 2;
    }
    return twice(3) + 1;
}
int main(void) { return outer(); }
