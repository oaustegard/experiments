/* Read graph6 from stdin. Emit lines whose graph has NO cycle of any length
   given on the command line.  Exact test, no sampling.
   Each cycle is searched from its minimum vertex, so each is reachable once. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int n;
static uint64_t adj[64];
static int target;

static int dfs(int s, int v, uint64_t seen, int len) {
    if (len == target) return (adj[v] >> s) & 1ULL;
    /* candidates: neighbours of v, unvisited, and > s */
    uint64_t cand = adj[v] & ~seen;
    cand &= ~((s == 63) ? ~0ULL : ((1ULL << (s + 1)) - 1ULL));
    while (cand) {
        int w = __builtin_ctzll(cand);
        cand &= cand - 1;
        if (dfs(s, w, seen | (1ULL << w), len + 1)) return 1;
    }
    return 0;
}

static int has_cycle(int L) {
    if (L > n) return 0;
    target = L;
    for (int s = 0; s + L <= n + 0; s++) {          /* need L distinct verts >= s */
        if (n - s < L) break;
        if (dfs(s, s, 1ULL << s, 1)) return 1;
    }
    return 0;
}

static int decode(const char *p) {
    n = p[0] - 63;
    if (n < 1 || n > 62) return 0;
    for (int i = 0; i < n; i++) adj[i] = 0;
    int k = 0;
    for (int j = 1; j < n; j++)
        for (int i = 0; i < j; i++) {
            int b = (p[1 + k / 6] - 63) >> (5 - k % 6) & 1;
            if (b) { adj[i] |= 1ULL << j; adj[j] |= 1ULL << i; }
            k++;
        }
    return 1;
}

int main(int argc, char **argv) {
    int L[8], nL = 0;
    for (int i = 1; i < argc && nL < 8; i++) L[nL++] = atoi(argv[i]);
    char line[4096];
    long long seen = 0, kept = 0;
    while (fgets(line, sizeof line, stdin)) {
        size_t len = strlen(line);
        while (len && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (!len) continue;
        if (!decode(line)) continue;
        seen++;
        int ok = 1;
        for (int i = 0; i < nL; i++) if (has_cycle(L[i])) { ok = 0; break; }
        if (ok) { kept++; printf("%s\n", line); fflush(stdout); }
    }
    fprintf(stderr, "scanned=%lld avoiding=%lld\n", seen, kept);
    return 0;
}
