// EXPECT-EXIT: 0
int main(void) {
    (int){3} = 5;
    int *p = &(int){1};
    *p += 2;
    return (int){0} == 0 && *p == 3 ? 0 : 1;
}
