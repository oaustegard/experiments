// EXPECT-ERROR: cast specifies array type
int main(void) {
    void *p = 0;
    (int []) p;
    return 0;
}
