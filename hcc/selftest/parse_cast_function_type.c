// EXPECT-ERROR: cast specifies function type
int main(void) {
    void *p = 0;
    (int ()) p;
    return 0;
}
