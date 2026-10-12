// EXPECT-EXIT: 0
// A sole unnamed parameter whose type is void (via typeof) means (void).
void *vp;
int main(typeof(*vp)) { return 0; }
