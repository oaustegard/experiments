// EXPECT-EXIT: 0
int sub(int i, int array[i++]) { return i; }
int main(void) {
    int array[10];
    return sub(10, array) != 11;
}
