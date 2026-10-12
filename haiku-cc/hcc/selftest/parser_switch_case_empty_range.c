// EXPECT-EXIT: 7
/* empty ranges (lo > hi) match nothing and do not conflict with other labels. */
int main(void) {
    int x = 7;
    switch (x) {
    case -14 ... -15:
        return 1;
    case -14 ... -15:
        return 2;
    case 7:
        return 7;
    }
    return 0;
}
