/* Headless doomgeneric backend for the hcc `doom` suite: no window, no input, a fake clock.
 *
 * The game is advanced for DOOM_TICS calls of doomgeneric_Tick() (environment, default 300).  Every
 * DG_DrawFrame() folds the whole frame buffer into an FNV-1a hash; the running hash, the frame count and the
 * game state are printed every 50 frames and at the end, so a build that renders or simulates anything
 * differently from the gcc build prints different lines.
 *
 * The first HASH_FROM frames are not hashed: they are the title page and the screen wipe into the level, and
 * the TITLEPIC drawing reads memory it does not own (a few pixel rows differ from run to run with ASLR, also
 * with the gcc build).  From the first frame after the wipe on, every frame is reproducible.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "doomgeneric.h"

#define HASH_FROM 28

static uint32_t fake_ms;
static uint64_t hash = 1469598103934665603ULL;
static unsigned long frames;

void DG_Init(void) {}

void DG_DrawFrame(void)
{
    const unsigned char *p = (const unsigned char *)DG_ScreenBuffer;
    size_t n = (size_t)DOOMGENERIC_RESX * DOOMGENERIC_RESY * sizeof(pixel_t);
    size_t i;
    if (frames >= HASH_FROM) {
        for (i = 0; i < n; i++) {
            hash ^= p[i];
            hash *= 1099511628211ULL;
        }
    }
    frames++;
    if (frames % 50 == 0)
        printf("frame %lu hash %016llx\n", frames, (unsigned long long)hash);
}

void DG_SleepMs(uint32_t ms) { fake_ms += ms; }

uint32_t DG_GetTicksMs(void)
{
    fake_ms += 3;
    return fake_ms;
}

int DG_GetKey(int *pressed, unsigned char *key)
{
    (void)pressed;
    (void)key;
    return 0;
}

void DG_SetWindowTitle(const char *title) { (void)title; }

int main(int argc, char **argv)
{
    const char *t = getenv("DOOM_TICS");
    long ticks = t ? atol(t) : 300, i;

    doomgeneric_Create(argc, argv);
    for (i = 0; i < ticks; i++)
        doomgeneric_Tick();
    printf("done frames %lu hash %016llx clock %u\n", frames, (unsigned long long)hash, (unsigned)fake_ms);
    return 0;
}
