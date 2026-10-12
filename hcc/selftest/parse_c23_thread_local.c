// EXPECT-EXIT: 0
// thread_local is a plain identifier outside C23; a user macro for it must work.
#define thread_local _Thread_local
static thread_local int counter = 3;
int main (void) { return counter - 3; }
