// EXPECT-OK
// EXPECT-FLAGS: -std=c23
// nullptr compares with pointers, nullptr and the integer constant 0.
int *p;
int main(void) {
    if (nullptr != 0 || p != nullptr || nullptr != nullptr || 0 != nullptr) return 1;
    return 0;
}
