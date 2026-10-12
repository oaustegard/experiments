"""Driver for the `doom` suite: doomgeneric (ozkl/doomgeneric, commit-pinned) built by hcc and run headless on
Freedoom (phase 1, v0.13.0, sha256-pinned) with a fake clock.

tests/suites/doom/doomgeneric_headless.c is the platform backend: no window, no input, deterministic clock; it
hashes every frame buffer and prints the running hash (see the file).  The reference is the same source built
by gcc -O0.  Everything printed (init messages, frame hashes, final state) and the exit status must be equal.

Test ids:
  doom/src/<u>.c      hcc compiles that one translation unit (79 doomgeneric files + the backend); the gcc -O0
                      objects of the rest are linked in; demo1 is played for 60 tics.
  doom/demo/demoN     entirely hcc-built, -timedemo demoN, 1000 tics (demo1: 800; demo4 is skipped: the gcc build itself
                      is not reproducible, see skip.txt).
  doom/warp/<map>     entirely hcc-built, `-warp E M -skill 3` with an idle player, 400 tics (different maps
                      exercise different parts of the renderer and play simulation).
"""
import os

import common as C
import projlib as P

URL = "https://github.com/ozkl/doomgeneric.git"
REF = "master"
COMMIT = "e257a09527062bfb986b6e5e0780839b6cbd5b71"
WAD_URL = "https://github.com/freedoom/freedoom/releases/download/v0.13.0/freedoom-0.13.0.zip"
WAD_SHA = "3f9b264f3e3ce503b4fb7f6bdcb1f419d93c7b546f4df3e874dd878db9688f59"
UNITS = """dummy am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop d_main d_mode d_net f_finale f_wipe
g_game hu_lib hu_stuff info i_cdmus i_endoom i_joystick i_scale i_sound i_system i_timer memio m_argv m_bbox m_cheat
m_config m_controls m_fixed m_menu m_misc m_random p_ceilng p_doors p_enemy p_floor p_inter p_lights p_map p_maputl
p_mobj p_plats p_pspr p_saveg p_setup p_sight p_spec p_switch p_telept p_tick p_user r_bsp r_data r_draw r_main
r_plane r_segs r_sky r_things sha1 sounds statdump st_lib st_stuff s_sound tables v_video wi_stuff w_checksum w_file
w_main w_wad z_zone w_file_stdc i_input i_video doomgeneric""".split()
BACKEND = "doomgeneric_headless"
WARPS = {"e1m1": "1 1", "e1m4": "1 4", "e1m8": "1 8", "e2m2": "2 2", "e2m6": "2 6", "e3m3": "3 3", "e3m8": "3 8",
         "e4m2": "4 2", "e4m5": "4 5"}
RUN_TIMEOUT = 240
DEFS = ["-DNORMALUNIX", "-DLINUX", "-DSNDSERV", "-D_DEFAULT_SOURCE"]


def list_tests(suite):
    return (["src/%s.c" % u for u in UNITS + [BACKEND]] + ["demo/demo%d" % i for i in (1, 2, 3, 4)]
            + ["warp/%s" % w for w in WARPS])


def project(R, suite, cc_argv):
    d = os.path.join(P.fetch_git("doomgeneric", URL, REF, COMMIT), "doomgeneric")
    units = {u: os.path.join(d, u + ".c") for u in UNITS}
    units[BACKEND] = os.path.join(suite.dir, BACKEND + ".c")
    return P.Project(R, suite, cc_argv, "doom", units, ["-I" + d] + DEFS, ["-lm"]), d


def wad():
    return os.path.join(P.fetch_tarball("freedoom", WAD_URL, WAD_SHA), "freedoom1.wad")


def play(R, exe, work, args, tics):
    env = dict(os.environ, HOME=work, DOOM_TICS=str(tics))
    r = C.run_and_capture(R, [exe, "-iwad", wad(), "-nosound", "-nomusic", "-nosfx"] + args, RUN_TIMEOUT, work, env=env)
    r["out"] = C.normalise(r["out"].replace(work, "<WORK>"), [(r"0x[0-9a-fA-F]+", "0xPTR")])
    r["err"] = r["err"][-500:]
    return r


def scenario(rel):
    kind, _, name = rel.partition("/")
    if kind == "src":
        return ["-timedemo", "demo1"], 60
    if kind == "demo":
        return ["-timedemo", name], 800 if name == "demo1" else 1000  # demo1 is only reproducible up to ~900 tics
    return ["-warp"] + WARPS[name].split() + ["-skill", "3"], 400


def run_test(suite, rel, cc_argv, work, R):
    proj, d = project(R, suite, cc_argv)
    kind, _, name = rel.partition("/")
    args, tics = scenario(rel)
    units = UNITS + [BACKEND]
    if kind == "src":
        exe = os.path.join(work, "doom-mixed")
        f = proj.mixed(exe, units, [name[:-2]], ["-lm"])
        if f:
            return f.result(R)
    else:
        exe, fail = proj.full("doom", units, ["-lm"])
        if fail:
            return fail.result(R)
    ref = proj.ref("doom", units, ["-lm"])
    want = P.ref_value("doom-ref", [C.file_sha(ref), C.file_sha(wad()), " ".join(args), str(tics), C.file_sha(__file__)],
                       lambda: play(R, ref, P.tmpdir(work, "ref"), args, tics))
    return C.compare_runs(R, play(R, exe, P.tmpdir(work, "got"), args, tics), want)
