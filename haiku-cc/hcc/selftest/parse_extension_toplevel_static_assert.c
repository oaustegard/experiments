// EXPECT-EXIT: 7
// __extension__ before a file-scope _Static_assert or declaration is a no-op marker.
__extension__ _Static_assert (1, "foo");
__extension__ [[]] int g = 3;
__extension__ int h = 4;
int main(void) {
    return g + h;
}
