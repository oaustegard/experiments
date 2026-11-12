// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * doom_host: PrBoom on Linux, played by a rollout planner.
 *
 * The game runs headless in single-tic mode: one game tic and one rendered
 * frame per loop iteration, with no wall clock involved. Every `commit` tics
 * the planner forks the whole process once per (candidate action, scenario).
 * Each child is an exact copy of the game; it reseeds the engine's random
 * number generator for its scenario, plays its candidate for `horizon` tics,
 * scores what happened from engine state, writes the score and its last frame
 * to shared memory, and exits. The parent keeps the candidate with the best
 * risk-adjusted score over its scenarios and plays it for `commit` tics,
 * recording every frame to ffmpeg.
 *
 * No scenario uses the real seed, so the planner never sees the actual
 * future, only plausible ones under the game's own physics and monster AI.
 *
 * When two candidates come out close, the parent can hand the choice to an
 * outside judge: it writes the leaders' end frames and prints a JUDGE line on
 * stdout, then reads "PICK <n>" from stdin (play.py answers with Haiku).
 */

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <execinfo.h>
#include <time.h>
#include <unistd.h>

#include <doomtype.h>
#include <doomstat.h>
#include <doomdef.h>
#include <d_main.h>
#include <d_event.h>
#include <g_game.h>
#include <i_system.h>
#include <i_video.h>
#include <i_sound.h>
#include <i_main.h>
#include <m_argv.h>
#include <m_random.h>
#include <r_draw.h>
#include <r_state.h>
#include <r_fps.h>
#include <s_sound.h>
#include <st_stuff.h>
#include <v_video.h>
#include <w_wad.h>
#include <z_zone.h>
#include <p_pspr.h>
#include <d_player.h>
#include <lprintf.h>

#define W 320
#define H 200
#define MAX_CAND 64
#define MAX_SCEN 8
#define MAX_SLOTS (MAX_CAND * MAX_SCEN)
#define NUMKEYS 512   // matches g_game.c

// ---- Options --------------------------------------------------------------

static struct {
    const char *out;       // output directory
    int horizon;           // tics each rollout plays
    int commit;            // tics the chosen candidate is played before re-planning
    int scenarios;         // perturbed seeds per candidate
    int max_tics;          // stop after this many committed tics
    int record;            // write video.mp4
    int judge;             // ask the outside judge on close calls
    double judge_margin;   // "close" = leader within this fraction of the runner-up
    int judge_every;       // at most one judge call per this many decisions
    int assist_exit;       // reward getting closer to an exit line
    double risk;           // 0 = mean over scenarios, 1 = worst case
    int skill;             // 1..5
    int episode, map;
} opt = {
    .out = "run", .horizon = 52, .commit = 8, .scenarios = 3, .max_tics = 35 * 300,
    .record = 1, .judge = 0, .judge_margin = 0.10, .judge_every = 4,
    .assist_exit = 0, .risk = 0.5, .skill = 3, .episode = 1, .map = 1,
};

// ---- Engine glue ----------------------------------------------------------

int snd_card = 0, mus_card = 0;
int snd_samplerate = 11025;
int current_palette = 0;

static uint8_t framebuffer[W * H];
static uint8_t palette_rgb[256 * 3];
static FILE *video;
static const char *doom_argv[32];

static int in_rollout;      // set in a forked child
static long committed_tics; // tics played in the real game
static long dealt_total;    // damage dealt by the player, from I_HapticHit

static long frames_written; // frames sent to ffmpeg
static int wipe_clock;      // see I_GetTime

void I_StartFrame(void) {}

// The screen-wipe loop calls this once per wipe frame; it is the only thing
// that moves I_GetTime's clock.
void I_UpdateNoBlit(void) { wipe_clock++; }
bool I_StartDisplay(void) { return true; }
void I_EndDisplay(void) {}
void I_UpdateVideoMode(void) {}
void I_ShutdownGraphics(void) {}
const char *I_SigString(char *buf, size_t sz, int signum)
{
    snprintf(buf, sz, "signal %d", signum);
    return buf;
}

static void frame_to_rgb(const uint8_t *src, uint8_t *rgb)
{
    for (int i = 0; i < W * H; i++)
        memcpy(rgb + i * 3, palette_rgb + src[i] * 3, 3);
}

void I_FinishUpdate(void)
{
    if (in_rollout || !video)
        return;
    frames_written++;
    static uint8_t rgb[W * H * 3];
    frame_to_rgb(framebuffer, rgb);
    fwrite(rgb, 1, sizeof(rgb), video);
}

void I_SetPalette(int pal)
{
    uint16_t *p = V_BuildPalette(pal, 16);
    for (int i = 0; i < 256; i++)
    {
        uint16_t c = p[i];
        palette_rgb[i * 3 + 0] = ((c >> 11) & 31) * 255 / 31;
        palette_rgb[i * 3 + 1] = ((c >> 5) & 63) * 255 / 63;
        palette_rgb[i * 3 + 2] = (c & 31) * 255 / 31;
    }
    Z_Free(p);
    current_palette = pal;
}

// Load every WAD the engine opened into memory and close the file. Forked
// rollouts would otherwise share one open file description, and so one file
// offset: two of them loading lumps at once race on fseek/fread and read
// each other's bytes. That corrupted patches (SIGSEGV in R_CachePatchNum)
// and made identical runs diverge. With wadfile->data set the engine reads
// lumps straight from memory. Runs after W_Init, before the first fork.
static void wads_to_memory(void)
{
    for (size_t i = 0; i < numwadfiles; i++)
    {
        wadfile_info_t *w = &wadfiles[i];
        if (w->data || !w->handle)
            continue;
        FILE *f = w->handle;
        void *buf = malloc(w->size);
        fseek(f, 0, SEEK_SET);
        if (!buf || fread(buf, 1, w->size, f) != w->size)
            I_Error("could not read %s into memory", w->name);
        fclose(f);
        w->handle = NULL;
        w->data = buf;
    }
}

void I_InitGraphics(void)
{
    wads_to_memory();
    for (int i = 0; i < 3; i++)
    {
        screens[i].width = SCREENWIDTH;
        screens[i].height = SCREENHEIGHT;
        screens[i].byte_pitch = SCREENWIDTH;
    }
    screens[0].data = framebuffer;
    screens[0].not_on_heap = true;
    screens[4].width = SCREENWIDTH;
    screens[4].height = (ST_SCALED_HEIGHT + 1);
    screens[4].byte_pitch = SCREENWIDTH;
}

// Game time, not wall time: the renderer's frame interpolation reads this,
// and a wall clock made runs differ (it shifts the view, which changes which
// lines get marked seen, which changes the planner's choices).
int I_GetTimeMS(void)
{
    return (int)((gametic + wipe_clock) * 1000L / 35);
}

// Game logic runs in single-tic mode and never needs this clock, but two
// engine paths read it. The screen wipe loops until it moves, so it advances
// once per wipe frame (I_UpdateNoBlit). D_Display calls D_BuildNewTiccmds,
// which calls I_StartTic once per tic this clock has advanced; a clock that
// moved on every call ran I_StartTic about 1.65 times per real tic, which
// double-counted tics and cut every rollout short. I_StartTic now also acts
// at most once per gametic.
int I_GetTime(void)
{
    return wipe_clock;
}

void I_uSleep(unsigned long usecs) {}

const char *I_DoomExeDir(void) { return opt.out; }

// Sound and music are off: the run is silent.
void I_InitSound(void) {}
void I_ShutdownSound(void) {}
void I_SetChannels(void) {}
int I_StartSound(int id, int channel, int vol, int sep, int pitch, int priority) { return -1; }
void I_StopSound(int handle) {}
bool I_SoundIsPlaying(int handle) { return false; }
bool I_AnySoundStillPlaying(void) { return false; }
void I_UpdateSoundParams(int handle, int vol, int sep, int pitch) {}
void I_InitMusic(void) {}
void I_ShutdownMusic(void) {}
void I_UpdateMusic(void) {}
void I_SetMusicVolume(int volume) {}
void I_PauseSong(int handle) {}
void I_ResumeSong(int handle) {}
int I_RegisterSong(const void *data, size_t len) { return 0; }
void I_PlaySong(int handle, int looping) {}
void I_StopSong(int handle) {}
void I_UnRegisterSong(int handle) {}

void I_HapticHit(int damage, int killed)
{
    dealt_total += damage;
}

// ---- Actions --------------------------------------------------------------

typedef struct {
    signed char fwd, turn, side;   // -1, 0, 1; turn 1 = right, side 1 = right
    unsigned char fire, use;
    unsigned char split;           // if set: after this many tics, run straight ahead
    char name[24];
} act_t;

static act_t cands[MAX_CAND];
static int ncands;

static void add_cand(int fwd, int turn, int side, int fire, int use)
{
    act_t *a = &cands[ncands++];
    a->fwd = fwd; a->turn = turn; a->side = side; a->fire = fire; a->use = use;
    char *p = a->name;
    p += sprintf(p, "%s", fwd > 0 ? "fwd" : fwd < 0 ? "back" : "");
    if (turn) p += sprintf(p, "%s%s", p > a->name ? "+" : "", turn > 0 ? "turnR" : "turnL");
    if (side) p += sprintf(p, "%s%s", p > a->name ? "+" : "", side > 0 ? "strafeR" : "strafeL");
    if (fire) p += sprintf(p, "%s%s", p > a->name ? "+" : "", "fire");
    if (use) p += sprintf(p, "%s%s", p > a->name ? "+" : "", "use");
    if (p == a->name) sprintf(p, "wait");
}

static void build_candidates(void)
{
    for (int fire = 0; fire <= 1; fire++)
        for (int fwd = 1; fwd >= -1; fwd--)
            for (int turn = -1; turn <= 1; turn++)
                add_cand(fwd, turn, 0, fire, 0);
    for (int side = -1; side <= 1; side += 2)
        for (int fire = 0; fire <= 1; fire++)
            add_cand(1, 0, side, fire, 0);
    add_cand(1, 0, 0, 0, 1);    // walk into a door or switch and press use
    add_cand(0, 0, 0, 0, 1);    // press use in place

    // Turn-then-go: face a new direction, then run. At 800 angle units per
    // tic, 20 tics is about 88 degrees and 41 tics about 180.
    static const struct { int turn, split; const char *name; } tg[] = {
        {-1, 20, "turnL90>fwd+use"}, {1, 20, "turnR90>fwd+use"},
        {-1, 10, "turnL45>fwd+use"}, {1, 10, "turnR45>fwd+use"}, {1, 41, "turn180>fwd+use"},
    };
    for (int i = 0; i < (int)(sizeof(tg) / sizeof(tg[0])); i++)
    {
        act_t *a = &cands[ncands++];
        memset(a, 0, sizeof(*a));
        a->turn = tg[i].turn;
        a->split = tg[i].split;
        snprintf(a->name, sizeof(a->name), "%s", tg[i].name);
    }
}

static bool keystate[NUMKEYS];

static void set_key(int key, bool down)
{
    if (key <= 0 || key >= NUMKEYS || keystate[key] == down)
        return;
    keystate[key] = down;
    event_t ev = {0};
    ev.type = down ? ev_keydown : ev_keyup;
    ev.data1 = key;
    D_PostEvent(&ev);
}

static void apply_action(const act_t *a, int t)
{
    if (a->split && t >= a->split)
    {
        // Run straight and pulse use, so facing a door or switch opens it
        static const act_t straight = {1, 0, 0, 0, 1, 0, "fwd+use"};
        a = &straight;
    }
    analog_forward = a->fwd * 50;          // forwardmove at run speed
    analog_turn = -a->turn * 800;          // angleturn units per tic; + is left
    set_key(key_strafeleft, a->side < 0);
    set_key(key_straferight, a->side > 0);
    set_key(key_fire, a->fire);
    set_key(key_use, a->use && (t % 6) < 2); // use fires on press, so pulse it
}

static void release_all(void)
{
    analog_forward = analog_turn = 0;
    set_key(key_strafeleft, false);
    set_key(key_straferight, false);
    set_key(key_fire, false);
    set_key(key_use, false);
}

// ---- Measurements ---------------------------------------------------------

#define CELL_SHIFT 7                   // 128-unit cells
#define CELLS 512                      // covers -32768..32767 map units
static uint8_t visited[CELLS * CELLS / 8];

static int cell_index(fixed_t x, fixed_t y)
{
    int cx = ((x >> FRACBITS) + 32768) >> CELL_SHIFT;
    int cy = ((y >> FRACBITS) + 32768) >> CELL_SHIFT;
    if (cx < 0 || cy < 0 || cx >= CELLS || cy >= CELLS)
        return -1;
    return cy * CELLS + cx;
}

// Lines of every sector the player has stood in. They count as explored even
// if never drawn: a portal line between sectors of equal height has no wall
// texture, so the renderer never marks it ML_MAPPED, and without this it
// stays the "nearest unseen line" forever and pins the player beside it.
static uint8_t *touched;
static uint8_t *switch_given_up;   // per line, reset with the level
static int touched_lines = -1, touched_map = -1;

static void touch_sector(void)
{
    mobj_t *mo = players[consoleplayer].mo;
    if (!mo || !mo->subsector)
        return;
    if (touched_lines != numlines || touched_map != gameepisode * 100 + gamemap)
    {
        free(touched);
        touched = calloc(numlines ? numlines : 1, 1);
        free(switch_given_up);
        switch_given_up = calloc(numlines ? numlines : 1, 1);
        touched_lines = numlines;
        touched_map = gameepisode * 100 + gamemap;
    }
    sector_t *sec = mo->subsector->sector;
    for (int k = 0; k < sec->linecount; k++)
        touched[sec->lines[k] - lines] = 1;
    // Anything within 128 units of where the player has been counts too
    double px = mo->x / 65536.0, py = mo->y / 65536.0;
    for (int i = 0; i < numlines; i++)
    {
        if (touched[i])
            continue;
        double mx = (lines[i].v1->x + lines[i].v2->x) / 131072.0;
        double my = (lines[i].v1->y + lines[i].v2->y) / 131072.0;
        if (fabs(mx - px) < 128 && fabs(my - py) < 128)
            touched[i] = 1;
    }
}

// Marks the player's cell visited; returns 1 if it was new.
static int visit(void)
{
    touch_sector();
    mobj_t *mo = players[consoleplayer].mo;
    if (!mo)
        return 0;
    int i = cell_index(mo->x, mo->y);
    if (i < 0 || (visited[i >> 3] & (1 << (i & 7))))
        return 0;
    visited[i >> 3] |= 1 << (i & 7);
    return 1;
}

static int mapped_lines(void)
{
    int n = 0;
    for (int i = 0; i < numlines; i++)
        n += (lines[i].flags & ML_MAPPED) != 0;
    return n;
}

static bool is_exit_special(int s)
{
    return s == 11 || s == 51 || s == 52 || s == 124 || s == 197 || s == 198;
}

static bool is_door_special(int s)
{
    // Manual doors (use to open): normal, keyed, stay-open, keyed stay-open
    return s == 1 || s == 26 || s == 27 || s == 28 || s == 31 || s == 32 ||
           s == 33 || s == 34 || s == 117 || s == 118;
}

// Whether a player can cross a two-sided line: a manual door, or an opening
// at least 56 units tall with a step of at most 24. Drop-offs count both
// ways, so a path may assume a ledge it cannot climb back up.
static bool passable(const line_t *l)
{
    if (!l->backsector || (l->flags & ML_BLOCKING))
        return false;
    if (is_door_special(l->special))
        return true;
    const sector_t *a = l->frontsector, *b = l->backsector;
    fixed_t floor = a->floorheight > b->floorheight ? a->floorheight : b->floorheight;
    fixed_t ceil = a->ceilingheight < b->ceilingheight ? a->ceilingheight : b->ceilingheight;
    fixed_t step = a->floorheight - b->floorheight;
    if (step < 0) step = -step;
    return ceil - floor >= 56 * FRACUNIT && step <= 24 * FRACUNIT;
}

// A line the renderer can draw: one-sided, or two-sided with a height
// difference (a step or a lintel) or a middle texture.
static bool is_visible_wall(const line_t *l)
{
    if (!l->backsector)
        return true;
    const sector_t *a = l->frontsector, *b = l->backsector;
    return a->floorheight != b->floorheight || a->ceilingheight != b->ceilingheight ||
           sides[l->sidenum[0]].midtexture || (l->sidenum[1] != NO_INDEX && sides[l->sidenum[1]].midtexture);
}

// Vanilla switch specials (S1 and SR): pressed with use on a wall. An S1
// switch clears its special once used, so "special still set" means untried.
static bool is_switch_special(int s)
{
    static const short sw[] = {7, 9, 11, 14, 15, 18, 20, 21, 23, 29, 41, 42, 43, 45, 49, 51,
                               55, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 99, 101,
                               102, 103, 111, 112, 113, 114, 115, 116, 122, 123, 127, 131,
                               132, 133, 134, 135, 136, 137, 138, 139, 140};
    for (size_t i = 0; i < sizeof(sw) / sizeof(sw[0]); i++)
        if (sw[i] == s)
            return true;
    return false;
}


// Target kinds for path_to: 0 = unseen wall, 1 = exit line (assist mode),
// 2 = untried switch (the fallback once nothing unseen is reachable)
static bool is_target(const line_t *l, int want_exit)
{
    if (want_exit == 2)
        return is_switch_special(l->special) && !l->backsector &&
               !(switch_given_up && touched_lines == numlines && switch_given_up[l - lines]);
    if (want_exit)
        return is_exit_special(l->special);
    if (l->flags & ML_MAPPED)
        return false;
    if (touched && touched_lines == numlines && touched[l - lines])
        return false;
    return is_visible_wall(l);
}

// Shortest walking distance, in map units, from the player to the nearest
// target line (an unseen line, or an exit line), moving through the sector
// graph from line midpoint to line midpoint. -1 when nothing is reachable.
// Dijkstra with an O(n^2) scan: E1M1 has under 500 lines.
static int path_target = -1;   // line index of the last path_to() result

static double path_to(int want_exit)
{
    path_target = -1;
    mobj_t *mo = players[consoleplayer].mo;
    if (!mo || !mo->subsector)
        return -1;
    static double *dist;
    static unsigned char *done;
    static int cap;
    if (cap < numlines)
    {
        free(dist);
        free(done);
        cap = numlines;
        dist = malloc(sizeof(double) * cap);
        done = malloc(cap);
    }
    double px = mo->x / 65536.0, py = mo->y / 65536.0;
#define MIDX(l) (((l)->v1->x + (l)->v2->x) / 131072.0)
#define MIDY(l) (((l)->v1->y + (l)->v2->y) / 131072.0)
    for (int i = 0; i < numlines; i++)
    {
        dist[i] = 1e18;
        done[i] = 0;
    }
    sector_t *start = mo->subsector->sector;
    for (int k = 0; k < start->linecount; k++)
    {
        line_t *l = start->lines[k];
        dist[l - lines] = hypot(MIDX(l) - px, MIDY(l) - py);
    }
    for (;;)
    {
        int u = -1;
        for (int i = 0; i < numlines; i++)
            if (!done[i] && dist[i] < 1e17 && (u < 0 || dist[i] < dist[u]))
                u = i;
        if (u < 0)
            return -1;
        done[u] = 1;
        line_t *l = &lines[u];
        if (is_target(l, want_exit))
        {
            path_target = u;
            return dist[u];
        }
        if (!passable(l))
            continue;
        sector_t *sides[2] = {l->frontsector, l->backsector};
        for (int s = 0; s < 2; s++)
            for (int k = 0; k < sides[s]->linecount; k++)
            {
                line_t *m = sides[s]->lines[k];
                int v = m - lines;
                double d = dist[u] + hypot(MIDX(m) - MIDX(l), MIDY(m) - MIDY(l));
                if (d < dist[v])
                    dist[v] = d;
            }
    }
#undef MIDX
#undef MIDY
}

typedef struct {
    fixed_t x, y;
    int health_armor, kills, items, mapped, ammo;
    long dealt;
    double frontier, exit_dist;
    int frontier_kind;   // 0 unseen walls, 2 untried switches
} snap_t;

static snap_t take_snapshot(void)
{
    player_t *p = &players[consoleplayer];
    snap_t s = {0};
    if (p->mo)
    {
        s.x = p->mo->x;
        s.y = p->mo->y;
    }
    s.health_armor = p->health + p->armorpoints;
    s.kills = p->killcount;
    s.items = p->itemcount;
    s.mapped = mapped_lines();
    for (int i = 0; i < NUMAMMO; i++)
        s.ammo += p->ammo[i];
    s.dealt = dealt_total;
    s.frontier = path_to(0);
    s.frontier_kind = 0;
    if (s.frontier < 0)
    {
        s.frontier = path_to(2);
        s.frontier_kind = 2;
    }
    s.exit_dist = opt.assist_exit ? path_to(1) : -1;
    return s;
}

typedef struct {
    double score;
    int dealt, kills, hurt, dead, explored, cells, items, ammo_used, exited, exit_tic;
    double frontier_gain, exit_gain, moved;
} result_t;

static double score_of(const result_t *r)
{
    return 12.0 * r->dealt + 150.0 * r->kills - 10.0 * r->hurt - 3000.0 * r->dead
         + 4.0 * r->explored + 25.0 * r->cells + 40.0 * r->items - 2.0 * r->ammo_used
         + (r->exited ? 10000.0 + 50.0 * (opt.horizon - r->exit_tic) : 0.0)
         + 0.3 * r->frontier_gain + 1.0 * r->exit_gain
         - (r->moved < 64 ? 20.0 : 0.0);   // standing still is not free
}

// ---- Planner --------------------------------------------------------------

static result_t *shm_res;       // [MAX_SLOTS]
static uint8_t *shm_frames;     // [MAX_SLOTS][W*H]

static snap_t ro_base;
static int ro_slot, ro_tic, ro_cells, ro_off, ro_exit_tic = -1;
static const act_t *ro_act;

// The plan being played and how many tics into it the real game is. Each
// decision also scores "continue": this plan resumed from cur_offset, so a
// turn-then-go plan is not restarted (and re-turned) every 8 tics.
static const act_t *current;
static int cur_offset;
static int plan_reaches_exit;   // every scenario of the chosen plan exits
static int commit_left;
#define CONTINUE_BONUS 10.0

// This decision's options: the fixed candidates plus "continue"
static const act_t *opt_act[MAX_CAND];
static int opt_off[MAX_CAND];
static char opt_name[MAX_CAND][40];
static int nopts;
static long decisions, judge_calls, last_judge_decision = -1000;
static FILE *dlog;

static void reseed(int scenario)
{
    for (int i = 0; i < NUMPRCLASS; i++)
        rng.seed[i] = rng.seed[i] * 1103515245ul + 12345ul * (unsigned long)scenario + i * 7919ul;
    rng.rndindex = (rng.rndindex + 37 * scenario) & 255;
    rng.prndindex = (rng.prndindex + 91 * scenario) & 255;
}

static void finish_rollout(void)
{
    player_t *p = &players[consoleplayer];
    snap_t end = take_snapshot();
    result_t r = {0};
    r.dealt = (int)(end.dealt - ro_base.dealt);
    r.kills = end.kills - ro_base.kills;
    r.hurt = ro_base.health_armor - end.health_armor;
    if (r.hurt < 0) r.hurt = 0;
    r.dead = p->playerstate == PST_DEAD;
    r.explored = end.mapped - ro_base.mapped;
    r.cells = ro_cells;
    r.items = end.items - ro_base.items;
    r.ammo_used = ro_base.ammo - end.ammo;
    if (r.ammo_used < 0) r.ammo_used = 0;
    r.exited = ro_exit_tic >= 0;
    r.exit_tic = r.exited ? ro_exit_tic : opt.horizon;
    if (ro_base.frontier_kind != end.frontier_kind)
        end.frontier = path_to(ro_base.frontier_kind);
    if (ro_base.frontier >= 0 && end.frontier >= 0)
        r.frontier_gain = ro_base.frontier - end.frontier;
    if (ro_base.exit_dist >= 0 && end.exit_dist >= 0)
        r.exit_gain = ro_base.exit_dist - end.exit_dist;
    r.moved = hypot((end.x - ro_base.x) / 65536.0, (end.y - ro_base.y) / 65536.0);
    r.score = score_of(&r);
    shm_res[ro_slot] = r;
    memcpy(shm_frames + (size_t)ro_slot * W * H, framebuffer, W * H);
    _exit(0);
}

static void write_ppm(const char *path, const uint8_t *frame)
{
    static uint8_t rgb[W * H * 3];
    frame_to_rgb(frame, rgb);
    FILE *f = fopen(path, "wb");
    if (!f)
        return;
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(rgb, 1, sizeof(rgb), f);
    fclose(f);
}

typedef struct {
    int cand;
    double agg, mean, worst;
} ranked_t;

static int cmp_ranked(const void *a, const void *b)
{
    double d = ((const ranked_t *)b)->agg - ((const ranked_t *)a)->agg;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

// Asks play.py to choose among the leaders. Returns a candidate index, or -1
// to keep the planner's choice.
static int ask_judge(const ranked_t *rk, int n, int nscen)
{
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/judge/%06ld", opt.out, decisions);
    mkdir(dir, 0755);
    printf("JUDGE {\"decision\": %ld, \"tic\": %ld, \"dir\": \"%s\", \"options\": [",
           decisions, committed_tics, dir);
    for (int i = 0; i < n; i++)
    {
        int c = rk[i].cand;
        // The frame from the scenario closest to the candidate's mean
        int best_s = 0;
        double best_d = 1e18;
        for (int s = 0; s < nscen; s++)
        {
            double d = fabs(shm_res[c * nscen + s].score - rk[i].mean);
            if (d < best_d) { best_d = d; best_s = s; }
        }
        char path[600];
        snprintf(path, sizeof(path), "%s/%c.ppm", dir, 'A' + i);
        write_ppm(path, shm_frames + (size_t)(c * nscen + best_s) * W * H);
        const result_t *r = &shm_res[c * nscen + best_s];
        printf("%s{\"label\": \"%c\", \"action\": \"%s\", \"score\": %.1f, \"frame\": \"%s\", "
               "\"dealt\": %d, \"hurt\": %d, \"explored\": %d, \"cells\": %d}",
               i ? ", " : "", 'A' + i, opt_name[c], rk[i].agg, path,
               r->dealt, r->hurt, r->explored, r->cells);
    }
    char now[600];
    snprintf(now, sizeof(now), "%s/now.ppm", dir);
    write_ppm(now, framebuffer);
    printf("], \"now\": \"%s\"}\n", now);
    fflush(stdout);

    char line[256];
    if (!fgets(line, sizeof(line), stdin))
        return -1;
    int pick;
    if (sscanf(line, "PICK %d", &pick) == 1 && pick >= 0 && pick < n)
        return rk[pick].cand;
    return -1;
}

static void crash_handler(int sig)
{
    void *bt[32];
    int n = backtrace(bt, 32);
    char msg[64];
    int len = snprintf(msg, sizeof(msg), "CRASH signal %d in %s\n", sig, in_rollout ? "rollout" : "game");
    write(2, msg, len);
    backtrace_symbols_fd(bt, n, 2);
    _exit(128 + sig);
}

static void decide(void)
{
    nopts = 0;
    for (int c = 0; c < ncands; c++)
    {
        opt_act[nopts] = &cands[c];
        opt_off[nopts] = 0;
        snprintf(opt_name[nopts], sizeof(opt_name[0]), "%s", cands[c].name);
        nopts++;
    }
    int cont = -1;
    if (current)
    {
        cont = nopts;
        opt_act[nopts] = current;
        opt_off[nopts] = cur_offset;
        snprintf(opt_name[nopts], sizeof(opt_name[0]), "continue:%s", current->name);
        nopts++;
    }

    int nscen = opt.scenarios, nslots = nopts * nscen;
    snap_t base = take_snapshot();
    pid_t pids[MAX_SLOTS];
    for (int i = 0; i < nslots; i++)
        shm_res[i].score = NAN;   // a child that never reports stays NaN

    fflush(NULL);
    for (int c = 0; c < nopts; c++)
        for (int s = 0; s < nscen; s++)
        {
            int slot = c * nscen + s;
            pid_t pid = fork();
            if (pid == 0)
            {
                in_rollout = 1;
                video = NULL;
                ro_base = base;
                ro_slot = slot;
                ro_tic = 0;
                ro_cells = 0;
                ro_exit_tic = -1;
                ro_act = opt_act[c];
                ro_off = opt_off[c];
                reseed(s + 1);   // scenario seeds 1..n; the real game is 0
                return;          // back into the game loop, now as a rollout
            }
            if (pid < 0)
            {
                perror("fork");
                exit(1);
            }
            pids[slot] = pid;
        }
    static int wstatus[MAX_SLOTS];
    for (int i = 0; i < nslots; i++)
        while (waitpid(pids[i], &wstatus[i], 0) < 0 && errno == EINTR)
            ;
    for (int i = 0; i < nslots; i++)
        if (isnan(shm_res[i].score))
        {
            fprintf(stderr, "rollout slot %d (%s) did not report: %s %d\n", i,
                    opt_name[i / nscen],
                    WIFSIGNALED(wstatus[i]) ? "signal" : "exit",
                    WIFSIGNALED(wstatus[i]) ? WTERMSIG(wstatus[i]) : WEXITSTATUS(wstatus[i]));
            exit(1);
        }

    ranked_t rk[MAX_CAND];
    for (int c = 0; c < nopts; c++)
    {
        double sum = 0, worst = 1e18;
        for (int s = 0; s < nscen; s++)
        {
            double v = shm_res[c * nscen + s].score;
            sum += v;
            if (v < worst) worst = v;
        }
        double mean = sum / nscen;
        double agg = mean - opt.risk * (mean - worst) + (c == cont ? CONTINUE_BONUS : 0);
        rk[c] = (ranked_t){c, agg, mean, worst};
    }
    qsort(rk, nopts, sizeof(rk[0]), cmp_ranked);

    int chosen = rk[0].cand;
    const char *by = "planner";
    double gap = rk[0].agg - rk[1].agg;
    double scale = fabs(rk[0].agg) > 1 ? fabs(rk[0].agg) : 1;
    if (opt.judge && gap / scale < opt.judge_margin &&
        decisions - last_judge_decision >= opt.judge_every)
    {
        int n = 4;
        while (n > 2 && rk[0].agg - rk[n - 1].agg > scale * 3 * opt.judge_margin)
            n--;
        int j = ask_judge(rk, n, nscen);
        judge_calls++;
        last_judge_decision = decisions;
        if (j >= 0)
        {
            chosen = j;
            by = "judge";
        }
    }
    plan_reaches_exit = 1;
    for (int sc = 0; sc < nscen; sc++)
        if (!shm_res[chosen * nscen + sc].exited)
            plan_reaches_exit = 0;
    if (chosen != cont)
    {
        current = opt_act[chosen];
        cur_offset = 0;
    }

    if (dlog)
    {
        fprintf(dlog, "{\"decision\": %ld, \"tic\": %ld, \"chosen\": \"%s\", \"by\": \"%s\", \"top\": [",
                decisions, committed_tics, opt_name[chosen], by);
        for (int i = 0; i < 5 && i < nopts; i++)
            fprintf(dlog, "%s[\"%s\", %.1f, %.1f]", i ? ", " : "", opt_name[rk[i].cand],
                    rk[i].agg, rk[i].worst);
        player_t *p = &players[consoleplayer];
        int tl = -1;
        double fd = path_to(0);
        tl = path_target;
        fprintf(dlog, "], \"health\": %d, \"armor\": %d, \"kills\": %d, \"mapped\": %d, \"map\": %d, "
                "\"pos\": [%d, %d], \"frontier\": %.0f, \"target\": %d",
                p->health, p->armorpoints, p->killcount, base.mapped, gamemap,
                p->mo ? p->mo->x >> FRACBITS : 0, p->mo ? p->mo->y >> FRACBITS : 0, fd, tl);
        if (tl >= 0)
        {
            line_t *l = &lines[tl];
            fprintf(dlog, ", \"target_line\": [%d, %d, %d, %d, %d, %d]",
                    l->v1->x >> FRACBITS, l->v1->y >> FRACBITS, l->v2->x >> FRACBITS,
                    l->v2->y >> FRACBITS, l->special, l->backsector != NULL);
        }
        fprintf(dlog, ", \"all\": {");
        for (int c = 0; c < nopts; c++)
        {
            fprintf(dlog, "%s\"%s\": [", c ? ", " : "", opt_name[c]);
            for (int sc = 0; sc < nscen; sc++)
                fprintf(dlog, "%s%.1f", sc ? ", " : "", shm_res[c * nscen + sc].score);
            fprintf(dlog, "]");
        }
        fprintf(dlog, "}}\n");
    }
    // A target that has stayed the nearest for 25 decisions (about 6 s)
    // without a single new line seen is dropped: something keeps the player
    // from it that the path search does not model.
    {
        static int last_target = -1, stuck_for = 0, last_mapped = -1;
        int kind = path_to(0) >= 0 ? 0 : 2;
        if (kind == 2 && path_to(2) < 0 && touched && touched_lines == numlines)
        {
            // Nothing reachable is left unseen and no switch is untried.
            // Forget the give-ups and proximity marks (keep what was actually
            // drawn) and look again.
            memset(touched, 0, numlines);
            memset(switch_given_up, 0, numlines);
            touch_sector();
            printf("RESET exploration at tic %ld\n", committed_tics);
            fflush(stdout);
            kind = path_to(0) >= 0 ? 0 : 2;
        }
        path_to(kind);
        if (path_target >= 0 && path_target == last_target && base.mapped == last_mapped)
        {
            if (++stuck_for >= 25 && kind == 2 && switch_given_up)
            {
                switch_given_up[path_target] = 1;
                stuck_for = 0;
                printf("GIVEUP switch %d at tic %ld\n", path_target, committed_tics);
                fflush(stdout);
            }
            else if (stuck_for >= 25 && touched && touched_lines == numlines)
            {
                // Drop the target and its neighbours: unreachable lines come
                // in clusters (a room seen through a window)
                line_t *t = &lines[path_target];
                double tx = (t->v1->x + t->v2->x) / 131072.0, ty = (t->v1->y + t->v2->y) / 131072.0;
                for (int i = 0; i < numlines; i++)
                {
                    double mx = (lines[i].v1->x + lines[i].v2->x) / 131072.0;
                    double my = (lines[i].v1->y + lines[i].v2->y) / 131072.0;
                    if (hypot(mx - tx, my - ty) < 192)
                        touched[i] = 1;
                }
                stuck_for = 0;
                printf("GIVEUP line %d at tic %ld\n", path_target, committed_tics);
                fflush(stdout);
            }
        }
        else
            stuck_for = 0;
        last_target = path_target;
        last_mapped = base.mapped;
    }
    decisions++;
}

// ---- Per-tic hook ---------------------------------------------------------

static long deaths, levels_done;
static int last_gamestate = -1, last_playerstate = -1;

static void write_summary(const char *why)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/summary.json", opt.out);
    FILE *f = fopen(path, "w");
    if (!f)
        return;
    player_t *p = &players[consoleplayer];
    fprintf(f, "{\"ended\": \"%s\", \"tics\": %ld, \"seconds\": %.1f, \"decisions\": %ld, "
               "\"judge_calls\": %ld, \"levels_completed\": %ld, \"deaths\": %ld, "
               "\"episode\": %d, \"map\": %d, \"health\": %d, \"kills\": %d, \"items\": %d, "
               "\"secrets\": %d, \"total_kills\": %d, \"total_items\": %d, \"total_secrets\": %d, "
               "\"damage_dealt\": %ld, \"candidates\": %d, \"scenarios\": %d, \"horizon\": %d, "
               "\"commit\": %d, \"risk\": %.2f, \"assist_exit\": %d, \"skill\": %d, "
               "\"frames\": %ld}\n",
            why, committed_tics, committed_tics / 35.0, decisions, judge_calls, levels_done, deaths,
            gameepisode, gamemap, p->health, p->killcount, p->itemcount, p->secretcount,
            totalkills, totalitems, totalsecret, dealt_total, ncands, opt.scenarios,
            opt.horizon, opt.commit, opt.risk, opt.assist_exit, opt.skill, frames_written);
    fclose(f);
}

static void finish(const char *why)
{
    write_summary(why);
    if (dlog)
        fclose(dlog);
    if (video)
        pclose(video);
    printf("END %s\n", why);
    fflush(stdout);
    exit(0);
}

void I_StartTic(void)
{
    static int acted_gametic = -1;
    if (gametic == acted_gametic)
        return;
    acted_gametic = gametic;

    if (in_rollout)
    {
        if (ro_exit_tic < 0 && (gamestate == GS_INTERMISSION || gameaction == ga_completed))
            ro_exit_tic = ro_tic;
        if (ro_tic == opt.horizon)
            finish_rollout();
        ro_cells += visit();
        apply_action(ro_act, ro_off + ro_tic);
        ro_tic++;
        return;
    }

    committed_tics++;
    if (committed_tics >= opt.max_tics)
        finish("max_tics");

    player_t *p = &players[consoleplayer];
    if (gamestate == GS_INTERMISSION && last_gamestate == GS_LEVEL)
    {
        levels_done++;
        printf("LEVEL %d done at tic %ld\n", gamemap, committed_tics);
        fflush(stdout);
    }
    if (p->playerstate == PST_DEAD && last_playerstate == PST_LIVE)
    {
        deaths++;
        printf("DIED on map %d at tic %ld\n", gamemap, committed_tics);
        fflush(stdout);
    }
    last_gamestate = gamestate;
    last_playerstate = p->playerstate;

    if (gamestate != GS_LEVEL || p->playerstate != PST_LIVE || menuactive)
    {
        // Intermission, finale or death: tap use to move on
        release_all();
        set_key(key_use, (committed_tics % 20) < 2);
        commit_left = 0;
        current = NULL;   // plans restart after a death or a level change
        return;
    }

    visit();
    if (commit_left <= 0)
    {
        decide();
        if (in_rollout)
        {
            // This process is now a rollout child; play its first tic
            ro_cells += visit();
            apply_action(ro_act, ro_off + ro_tic++);
            return;
        }
        commit_left = opt.commit;
        if (plan_reaches_exit)
            commit_left = opt.horizon;   // play it through: re-planning mid-plan restarts it
        if (decisions % 50 == 0)
        {
            printf("STATUS tic=%ld decisions=%ld map=%d health=%d kills=%d/%d mapped=%d\n",
                   committed_tics, decisions, gamemap, p->health, p->killcount, totalkills,
                   mapped_lines());
            fflush(stdout);
        }
    }
    apply_action(current, cur_offset++);
    commit_left--;
}

void I_SafeExit(int rc)
{
    finish("engine_exit");
}

void I_Init(void)
{
    snd_channels = 8;
    autorun = 1;
    movement_smooth = 0;   // no frame interpolation: one frame per tic
}

// ---- Main -----------------------------------------------------------------

static void usage(void)
{
    fprintf(stderr,
            "doom_host [--out DIR] [--horizon N] [--commit N] [--scenarios N] [--max-tics N]\n"
            "          [--no-record] [--judge] [--judge-margin F] [--judge-every N]\n"
            "          [--assist-exit] [--risk F] [--skill N] [--warp E M] --iwad PATH\n");
    exit(2);
}

int main(int argc, char **argv)
{
    const char *iwad = NULL;
    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];
#define NEXT (i + 1 < argc ? argv[++i] : (usage(), ""))
        if (!strcmp(a, "--out")) opt.out = NEXT;
        else if (!strcmp(a, "--horizon")) opt.horizon = atoi(NEXT);
        else if (!strcmp(a, "--commit")) opt.commit = atoi(NEXT);
        else if (!strcmp(a, "--scenarios")) opt.scenarios = atoi(NEXT);
        else if (!strcmp(a, "--max-tics")) opt.max_tics = atoi(NEXT);
        else if (!strcmp(a, "--no-record")) opt.record = 0;
        else if (!strcmp(a, "--judge")) opt.judge = 1;
        else if (!strcmp(a, "--judge-margin")) opt.judge_margin = atof(NEXT);
        else if (!strcmp(a, "--judge-every")) opt.judge_every = atoi(NEXT);
        else if (!strcmp(a, "--assist-exit")) opt.assist_exit = 1;
        else if (!strcmp(a, "--risk")) opt.risk = atof(NEXT);
        else if (!strcmp(a, "--skill")) opt.skill = atoi(NEXT);
        else if (!strcmp(a, "--warp")) { opt.episode = atoi(NEXT); opt.map = atoi(NEXT); }
        else if (!strcmp(a, "--iwad")) iwad = NEXT;
        else usage();
    }
    if (!iwad || opt.scenarios < 1 || opt.scenarios > MAX_SCEN || opt.horizon < 1 || opt.commit < 1)
        usage();

    // If SIGCHLD arrives ignored (inherited across exec), the kernel reaps
    // rollouts itself and waitpid returns at once: the parent then reads
    // slots no child has written yet. Measured: about a third of the slots.
    signal(SIGCHLD, SIG_DFL);
    signal(SIGSEGV, crash_handler);

    mkdir(opt.out, 0755);
    char path[600];
    snprintf(path, sizeof(path), "%s/judge", opt.out);
    mkdir(path, 0755);
    snprintf(path, sizeof(path), "%s/decisions.jsonl", opt.out);
    dlog = fopen(path, "w");

    build_candidates();
    shm_res = mmap(NULL, sizeof(result_t) * MAX_SLOTS, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    shm_frames = mmap(NULL, (size_t)W * H * MAX_SLOTS, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shm_res == MAP_FAILED || shm_frames == MAP_FAILED)
    {
        perror("mmap");
        return 1;
    }

    if (opt.record)
    {
        // 320x200 with non-square pixels, shown at 4:3 as Doom was
        char cmd[1200];
        snprintf(cmd, sizeof(cmd),
                 "/usr/bin/ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s %dx%d -r 35 -i - "
                 "-vf scale=640:480:flags=neighbor -c:v libx264 -preset veryfast -crf 18 "
                 "-pix_fmt yuv420p '%s/video.mp4'", W, H, opt.out);
        video = popen(cmd, "w");
    }

    static char ep[8], mp[8], sk[8];
    snprintf(ep, sizeof(ep), "%d", opt.episode);
    snprintf(mp, sizeof(mp), "%d", opt.map);
    snprintf(sk, sizeof(sk), "%d", opt.skill);
    int n = 0;
    doom_argv[n++] = "doom";
    doom_argv[n++] = "-iwad";
    doom_argv[n++] = iwad;
    doom_argv[n++] = "-skill";
    doom_argv[n++] = sk;
    doom_argv[n++] = "-warp";
    doom_argv[n++] = ep;
    doom_argv[n++] = mp;
    doom_argv[n] = NULL;
    myargv = doom_argv;
    myargc = n;

    SCREENWIDTH = W;
    SCREENHEIGHT = H;
    singletics = true;

    printf("START candidates=%d scenarios=%d horizon=%d commit=%d judge=%d\n",
           ncands, opt.scenarios, opt.horizon, opt.commit, opt.judge);
    fflush(stdout);

    Z_Init();
    D_DoomMain();
    return 0;
}
