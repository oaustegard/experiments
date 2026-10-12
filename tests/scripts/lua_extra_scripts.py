# Small extra Lua programs for the lua suite: tests/suites/lua/extra/<nn>_<slug>.lua.
# Run by the fully hcc-built lua, compared with a gcc-built lua (stdout + exit code).
# Deterministic only (no os.time/clock output, no random values, no addresses).

SCRIPTS = [
("integer_float_semantics", """
print(1, 1.0, -0.0, 3 // 2, 3.0 // 2, -3 // 2, 3 % -2, -3 % 2, 3.5 % 2, 2^2, 7 // 0.0, -7 // 0.0, 0/0 ~= 0/0)
print(math.maxinteger + 1 == math.mininteger, math.maxinteger * 2, math.mininteger // -1, pcall(function() return 1 // 0 end))
print(1e15, 1e16, 2^53, 2^63, 123456789012345678, 0.1 + 0.2, 100 // 1e-1, 3 | 0, pcall(function() return 3.5 | 0 end))
print(string.format("%d %s %g %.14g", 3, 3.0, 3.0, 0.1), tostring(2^31), 1 < 1.5, math.tointeger("8"), math.tointeger(2^53))
print(8 // 3 * 3 + 8 % 3, (-8) // 3 * 3 + (-8) % 3, 5.3 % -2, -5.3 % 2, 1e308 * 10 // 1, math.fmod(-5, 3), math.fmod(5, -3))
"""),
("string_format_all", """
print(string.format("%5d|%-5d|%05d|%+d|% d", 42, 42, 42, 42, 42))
print(string.format("%x|%X|%#x|%o|%c|%i", 255, 255, 255, 8, 65, 7))
print(string.format("%e|%E|%.3e|%g|%G|%a", 12345.6789, 12345.6789, 0.00012345, 1e20, 1e-10, 1.0))
print(string.format("%10.3f|%-10.2f|%.0f|%#.0f|%f", 3.14159, 2.5, 2.5, 2.5, 1/3))
print(string.format("%s|%10s|%-10s|%.2s|%q", "str", "right", "left", "truncate", 'q"uo\\te\\n'))
print(string.format("%q %q %q %q", 1, 1.5, math.mininteger, 1/0))
print(pcall(string.format, "%d", 1.5), pcall(string.format, "%z", 1), pcall(string.format, "%d"))
print(#string.format("%99d", 1), pcall(string.format, "%100d", 1))
"""),
("patterns_find_gsub", """
print(string.find("hello world", "o w"), string.find("hello", "l+"), string.find("a.b", ".", 1, true), string.find("abc", "b", -1))
print(string.match("2024-03-15", "(%d+)-(%d+)-(%d+)"), string.match("  trim  ", "^%s*(.-)%s*$") .. "|", string.match("key=val", "(%w+)=(%w+)"))
print(string.gsub("hello world", "o", "0"), string.gsub("abc", "%w", "%0%0"), string.gsub("abc", "", "-"), string.gsub("a b c", " ", "_", 1))
print(string.gsub("hello", "l", {l = "L"}), string.gsub("$name is $age", "%$(%w+)", {name = "Bob", age = 42}), string.gsub("x", "x", "%%"))
print(string.gsub("abc def", "(%w+) (%w+)", "%2 %1"), string.gsub("THE (quick) fox", "%((%w+)%)", "<%1>"), string.match("[[nested]]", "%[(%b[])%]"))
for w in string.gmatch("one two  three", "%a+") do io.write(w, ",") end print()
for k, v in string.gmatch("a=1, b=2", "(%w+)=(%w+)") do io.write(k, v, ";") end print()
print(pcall(string.find, "abc", "[a-"), pcall(string.rep, "x", 1e10), string.match("f(a,b)", "%((.-)%)"), string.match("x", "()"))
print(string.match("  42  ", "%d+"), string.match("0x1F", "^0[xX](%x+)$"), string.match("a.b.c", "^(.-)%."), string.match("a.b.c", "^(.*)%."), string.find("a\\0b", "\\0", 1, true))
print(("%f[%w]%w+"):len(), string.gsub("THE END", "%f[%a]%a+", string.lower), string.gsub("a,b,,c", ",", ";"))
"""),
("string_library_misc", """
print(("abc"):rep(3, "-"), ("x"):rep(0), ("x"):rep(-1), ("abc"):reverse(), ("MiXed"):upper(), ("MiXed"):lower(), ("abc"):len(), #"\\0\\0")
print(("hello"):sub(2, 3), ("hello"):sub(-3), ("hello"):sub(2), ("hello"):sub(0), ("hello"):sub(10), ("hello"):sub(-100, 2), ("hello"):sub(3, 2))
print(("A"):byte(), ("ABC"):byte(1, -1), string.char(72, 101, 108, 108, 111), pcall(string.char, 256), ("é"):byte(1, -1))
print(tostring(12), tostring(1.5), tostring(nil), tostring(true), tonumber("0x10"), tonumber("10", 2), tonumber("  5  "), tonumber("5x"), tonumber("1e1"), tonumber(""), tonumber("z", 36))
print(#"héllo", utf8.len("héllo"), utf8.char(233, 8364), utf8.codepoint("héllo", 1, -1))
print(table.concat({1, 2, 3}, ", "), table.concat({}, "x"), table.concat({"a"}, "x"), table.concat({1, 2, 3}, "-", 2, 3), pcall(table.concat, {{}}))
print(select("#"), select("#", nil, nil), select(2, "a", "b", "c"), select(-1, "a", "b", "c"), pcall(select, 0))
print(("x"):rep(3, ", "), ("%d items"):format(3), ("abc"):find("b", 1, true), #("x"):rep(1000), ("a,b"):find(",", 1, true))
"""),
("string_pack_unpack", """
local p = string.pack("<i4 >i4 =i2 z s1 B", 1, 2, 3, "zero", "pfx", 255)
print(#p, p:byte(1, 12))
print(string.unpack("<i4 >i4 =i2 z s1 B", p))
print(string.pack("<d", 1.5):byte(1, -1))
print(string.unpack("<d", string.pack("<d", -0.1)), string.unpack("<f", string.pack("<f", 0.5)), string.unpack("<n", string.pack("<n", 2^60)))
print(string.packsize("<i1 i2 i4 i8"), string.packsize("<!4 i1 i4"), string.packsize("<!8 i1 i8"), pcall(string.packsize, "s"))
print(string.unpack("<I3", "\\1\\2\\3"), string.unpack(">I3", "\\1\\2\\3"), string.unpack("<i3", "\\255\\255\\255"), pcall(string.pack, "i1", 200))
print(string.pack(">I2", 258):byte(1, -1), string.pack("<j", math.mininteger):byte(8), string.unpack("z", "abc\\0def"), string.unpack("c3", "abcdef"), string.pack("c5", "ab"):byte(1, -1))
print(pcall(string.unpack, "i4", "abc"), string.unpack("<i2", "\\1\\0\\2\\0", 3), string.pack("<i16", -2):byte(1, 16))
"""),
("table_functions", """
local t = {5, 2, 8, 1, 9, 3}
table.sort(t) print(table.concat(t, " "))
table.sort(t, function(a, b) return a > b end) print(table.concat(t, " "))
print(table.unpack({1, 2, 3})) print(table.unpack({1, 2, 3}, 2)) print(table.unpack({1, 2, 3}, 2, 5)) print(table.pack(1, nil, 3).n, #table.pack())
table.insert(t, 100) table.insert(t, 1, 0) print(table.concat(t, ","), table.remove(t), table.remove(t, 1), table.remove({}), #t)
print(table.concat(table.move({1, 2, 3}, 1, 3, 3), ","), table.concat(table.move({1, 2, 3}, 2, 3, 1), ","), table.concat(table.move({1, 2}, 1, 2, 1, {9, 9, 9}), ","))
print(pcall(table.insert, {1, 2}, 5, 1), pcall(table.insert, {}, 1, 2, 3), pcall(table.sort, {3, 2, "x"}))
local names = {"bob", "Alice", "carol", "Dave"} table.sort(names) print(table.concat(names, " ")) table.sort(names, function(a, b) return a:lower() < b:lower() end) print(table.concat(names, " "))
local big = {} for i = 1, 500 do big[i] = (i * 7919) % 1009 end table.sort(big) local ok = true for i = 2, #big do ok = ok and big[i - 1] <= big[i] end print(ok, big[1], big[500], #big)
print(#{1, 2, 3, nil, 5}, #{n = 3}, #{[1] = 1, [2] = 2}, next({}), rawlen({1, 2}), rawlen("abc"), rawequal("a", "a"), rawget(setmetatable({}, {__index = function() return 1 end}), "x"))
"""),
("metamethods_all", """
local mt = {}
mt.__add = function(a, b) return "add" end mt.__sub = function() return "sub" end mt.__mul = function() return "mul" end mt.__div = function() return "div" end
mt.__mod = function() return "mod" end mt.__pow = function() return "pow" end mt.__unm = function() return "unm" end mt.__idiv = function() return "idiv" end
mt.__band = function() return "band" end mt.__bor = function() return "bor" end mt.__bxor = function() return "bxor" end mt.__shl = function() return "shl" end mt.__shr = function() return "shr" end mt.__bnot = function() return "bnot" end
mt.__concat = function() return "concat" end mt.__len = function() return 42 end mt.__eq = function() return true end mt.__lt = function() return true end mt.__le = function() return false end
mt.__call = function(self, ...) return "call", ... end mt.__tostring = function() return "OBJ" end mt.__name = "MyType"
local a, b = setmetatable({}, mt), setmetatable({}, mt)
print(a + 1, 1 - a, a * a, a / 2, a % 2, a ^ 2, -a, a // 2, a & 1, 1 | a, a ~ 1, a << 1, a >> 1, ~a)
print(a .. "x", "x" .. a, #a, a == b, a ~= b, a < b, a <= b, a > b, a >= b, a(1, 2), tostring(a))
print(rawequal(a, b), a == 1, pcall(function() return a < 1 end))
local log = {} local p = setmetatable({}, {__index = function(t, k) log[#log + 1] = "get " .. k return k .. "!" end, __newindex = function(t, k, v) log[#log + 1] = "set " .. k rawset(t, k, v) end})
print(p.x, p.y) p.z = 1 p.z = 2 print(table.concat(log, ","), p.z)
local ro = setmetatable({}, {__newindex = function() error("readonly", 2) end}) print(pcall(function() ro.x = 1 end))
print(getmetatable("abc").__index == string, getmetatable(setmetatable({}, {__metatable = "locked"})), pcall(setmetatable, setmetatable({}, {__metatable = 1}), {}))
"""),
("inheritance_oop", """
local Animal = {} Animal.__index = Animal
function Animal.new(name, sound) return setmetatable({name = name, sound = sound}, Animal) end
function Animal:speak() return self.name .. " says " .. self.sound end
local Dog = setmetatable({}, {__index = Animal}) Dog.__index = Dog
function Dog.new(name) local d = Animal.new(name, "woof") return setmetatable(d, Dog) end
function Dog:fetch() return self.name .. " fetches" end
local d = Dog.new("Rex") print(d:speak(), d:fetch(), getmetatable(d) == Dog, rawget(Dog, "speak"), d.nothing)
local Stack = {} Stack.__index = Stack function Stack.new() return setmetatable({n = 0}, Stack) end
function Stack:push(v) self.n = self.n + 1 self[self.n] = v return self end function Stack:pop() local v = self[self.n] self[self.n] = nil self.n = self.n - 1 return v end
local s = Stack.new():push(1):push(2):push(3) print(s:pop(), s:pop(), s.n)
local Counter = setmetatable({}, {__call = function(cls, start) return setmetatable({v = start}, cls) end}) Counter.__index = Counter
function Counter:next() self.v = self.v + 1 return self.v end print(Counter(10):next(), Counter(0):next())
"""),
("closures_upvalues", """
local function counter() local n = 0 return function() n = n + 1 return n end, function() return n end end
local inc, get = counter() inc() inc() print(get())
local fns = {} for i = 1, 3 do fns[i] = function() return i end end print(fns[1](), fns[2](), fns[3]())
local acc = {} do local x = 0 for i = 1, 3 do local y = i * 10 acc[i] = function() x = x + y return x end end end print(acc[1](), acc[2](), acc[3](), acc[1]())
local function shared() local v = 0 return function() v = v + 1 return v end, function() v = v * 10 return v end end
local a, b = shared() a() a() print(b(), a())
local function deep(n) if n == 0 then return function() return "bottom" end end local f = deep(n - 1) return function() return f() end end print(deep(50)())
local t = {} for i = 1, 3 do local j = i t[#t + 1] = function() j = j + 1 return j end end print(t[1](), t[1](), t[2](), t[3]())
print(select("#", (function(...) return ... end)(1, nil, nil)), (function(...) local a, b = ... return b end)(1, 2))
"""),
("varargs_multiple_returns", """
local function f(...) return select("#", ...), ... end
print(f()) print(f(nil)) print(f(1, nil, 3, nil)) print((f(1, 2, 3)))
local function g() return 1, 2, 3 end
local t = {g(), g()} print(#t) t = {g(), (g())} print(#t) t = {(g())} print(#t) print(g(), 10) print(10, g()) print((g()))
local a, b, c, d = g() print(a, b, c, d) local x, y = (g()) print(x, y)
print(table.pack(g()).n, math.max(g()), string.format("%d %d %d", g()), #{n = 1, g()})
local function sum(...) local s = 0 for i = 1, select("#", ...) do s = s + (select(i, ...) or 0) end return s end print(sum(1, 2, nil, 4), sum())
local function tail(...) return ... end print(tail(), tail(nil, nil), (tail(5, 6)))
print(pcall(function(...) return ... end, 1, 2, 3))
"""),
("coroutines_deep", """
local function gen(n) return coroutine.wrap(function() for i = 1, n do coroutine.yield(i) end end) end
local s = 0 for v in gen(100) do s = s + v end print(s)
local co = coroutine.create(function(a, b) print("start", a, b) local c = coroutine.yield(a + b) print("got", c) local d, e = coroutine.yield(c * 2) print("got2", d, e) return "end", 99 end)
print(coroutine.resume(co, 1, 2)) print(coroutine.status(co)) print(coroutine.resume(co, 10)) print(coroutine.resume(co, "x", "y")) print(coroutine.status(co), coroutine.resume(co))
local function producer() return coroutine.create(function() for i = 1, 3 do coroutine.yield("item" .. i) end return nil end) end
local p = producer() while true do local ok, v = coroutine.resume(p) if not v then break end io.write(v, " ") end print(coroutine.status(p))
local nested = coroutine.wrap(function() local inner = coroutine.wrap(function() coroutine.yield(1) coroutine.yield(2) end) coroutine.yield(inner()) coroutine.yield(inner()) coroutine.yield(3) end) print(nested(), nested(), nested())
local co2 = coroutine.create(function() error("boom") end) print(coroutine.resume(co2)) print(coroutine.status(co2))
local co3 = coroutine.create(function() local x <close> = setmetatable({}, {__close = function() print("closed") end}) coroutine.yield(1) end) coroutine.resume(co3) print(coroutine.close(co3), coroutine.status(co3))
print(coroutine.isyieldable(), coroutine.wrap(function() return coroutine.isyieldable() end)(), select(2, coroutine.running()), pcall(coroutine.yield, 1))
local ws = {} for i = 1, 200 do ws[i] = coroutine.wrap(function() local k = i coroutine.yield(k) return k * 2 end) end local tot = 0 for i = 1, 200 do tot = tot + ws[i]() + ws[i]() end print(tot)
"""),
("error_handling", """
print(pcall(error)) print(pcall(error, nil)) print(pcall(error, "msg", 0)) print(pcall(error, {code = 1})) print(select("#", pcall(error)))
print(xpcall(function() error("E") end, function(m) return "H:" .. m end)) print(xpcall(function(a, b) return a + b end, print, 3, 4))
print(pcall(function() local x = nil; return x.y end)) print(pcall(function() return #nil end)) print(pcall(function() return {} < {} end)) print(pcall(function() return "a" + 1 end)) print(pcall(function() undefined_fn() end))
print(pcall(function() local t = setmetatable({}, {__index = function() error("idx") end}) return t.x end))
local ok, e = pcall(error, setmetatable({}, {__tostring = function() return "custom" end})) print(ok, tostring(e))
print(select(2, pcall(error, "lvl1", 1)), select(2, pcall(error, "lvl2", 2)), pcall(assert, false), pcall(assert, nil, "amsg"), pcall(assert, 1, 2), pcall(assert, false, {1}))
print(pcall(string.rep), pcall(("x").rep, "x", {}), pcall(setmetatable, 1), pcall(ipairs), pcall(tostring), pcall(math.floor, "a"), pcall(tonumber, {}, 10))
local function lvl() error("deep", 2) end local function caller() lvl() end print(pcall(caller))
print(pcall(load, nil), load("x ="), load("return 1", "=chunk", "t", {}), load("return +"), pcall(load, "return 1", "c", "b"))
local f = load("local a, b = ... return a * b") print(f(6, 7)) print(load("x = 1; return x", "env", "t", {})())
print(string.format("%q", select(2, load("for"))), select(2, load("x = = 1", "=name")), select(2, load("goto nolabel", "=g")))
"""),
("goto_and_control", """
for i = 1, 3 do for j = 1, 3 do if j == 2 then goto continue end io.write(i, ":", j, " ") ::continue:: end end print()
do local i = 1 ::top:: if i <= 3 then io.write(i, " ") i = i + 1 goto top end print() end
local function f(n) local r = {} for i = 1, n do if i % 2 == 0 then goto skip end r[#r + 1] = i ::skip:: end return table.concat(r, ",") end print(f(10))
print(load("goto x; local a; ::x:: print(a)")) print(load("::l:: ::l::")) print(load("do goto l end ::l::") ~= nil)
local i = 0 repeat local j = i i = i + 1 until j >= 3 print(i)
for i = 10, 1, -3 do io.write(i, " ") end for i = 1, 0 do io.write("never") end for i = 1.0, 2 do io.write(i, " ") end print(pcall(function() for i = 1, 10, 0 do end end))
for i = math.maxinteger - 2, math.maxinteger do io.write(i, " ") end print() for i = 3, 1 do print("no") end
local n = 0 while true do n = n + 1 if n > 5 then break end end print(n)
local t = {10, 20, 30, nil, 50} for i, v in ipairs(t) do io.write(i, "=", v, " ") end print() local c = 0 for k, v in pairs({a = 1, b = 2, 3}) do c = c + 1 end print(c)
for k, v in next, {x = 1} do print(k, v) end local function it(s, c) if c < s then return c + 1 end end for v in it, 3, 0 do io.write(v, " ") end print()
"""),
("gc_and_weak_tables", """
local weakk = setmetatable({}, {__mode = "k"}) local weakv = setmetatable({}, {__mode = "v"})
do local key = {} weakk[key] = 1 weakv[1] = {} weakv[2] = "string stays" weakv[3] = 42 end
collectgarbage() collectgarbage() local n = 0 for _ in pairs(weakk) do n = n + 1 end print(n, weakv[1], weakv[2], weakv[3])
local fin = {} do local obj = setmetatable({}, {__gc = function() fin[#fin + 1] = "finalized" end}) end collectgarbage() collectgarbage() print(#fin, fin[1])
print(type(collectgarbage("count")), collectgarbage("step") ~= nil, collectgarbage("isrunning"), collectgarbage("incremental"), collectgarbage("generational"), collectgarbage("incremental"))
local before = collectgarbage("count") local junk = {} for i = 1, 10000 do junk[i] = {i} end local mid = collectgarbage("count") junk = nil collectgarbage() local after = collectgarbage("count") print(mid > before, after < mid)
local cache = setmetatable({}, {__mode = "kv"}) for i = 1, 100 do cache[i] = {i} end collectgarbage() local alive = 0 for _ in pairs(cache) do alive = alive + 1 end print(alive)
local t = setmetatable({}, {__gc = function(o) print("gc at close") end}) t = nil
local big = {} for i = 1, 100 do big[i] = string.rep("x", 1000) .. i end print(#big, #big[100])
local ephem = setmetatable({}, {__mode = "k"}) do local k = {} ephem[k] = {ref = k} end collectgarbage() print(next(ephem))
"""),
("load_env_and_chunks", """
local env = {print = print, x = 5} local f = load("x = x + 1 return x", "chunk", "t", env) print(f(), f(), env.x, x)
local pieces, pi = {"return ", "1 + ", "2"}, 0 local g = load(function() pi = pi + 1 return pieces[pi] end) print(g and g())
print(pcall(load("error('in chunk')", "=mychunk"))) print(pcall(load("local x = nil; x()", "@file.lua")))
local dumped = string.dump(function(a, b) return a * b + 1 end) local h = load(dumped, "dumped", "b") print(h(6, 7), #dumped > 10, load(dumped, "d", "t"))
local up = 10 local function useup() return up end local d2 = load(string.dump(useup)) print(pcall(d2))
print(dofile == nil, loadstring, setfenv, unpack, table.getn, math.pow, string.gfind, _VERSION, _G._G == _G)
local s = setmetatable({}, {__index = _G}) local fn = load("y = 7 return print ~= nil", "e", "t", s) print(fn(), rawget(s, "y"), y)
print(select("#", load("return ...")(1, 2, 3)), load("return select('#', ...)")(nil, nil), (load("local a <const> = 5; return a"))())
print(load("local a <const> = 1; a = 2")) print(load("local a <close> = nil; return 1")(), pcall(load("local a <close> = 1")))
print(require == nil, package.path ~= nil, type(package.loaded), package.loaded.string == string, pcall(require, "nonexistent_module_xyz") == false)
"""),
("os_io_functions", """
print(type(os.time()), os.time{year=2000, month=1, day=1, hour=0} ~= nil, os.date("!%Y-%m-%d %H:%M:%S", 0), os.date("!%c", 0), os.date("!%x %X %p", 3600 * 15))
print(os.date("!*t", 86400).day, os.date("!*t", 0).isdst, os.date("!%j %a %A %b %B %d %y %Z", 86400 * 40), pcall(os.date, "%Ez"), pcall(os.date, "%Q"))
print(os.getenv("HOME") ~= nil or true, os.getenv("NO_SUCH_VAR_XYZ"), type(os.clock()), os.difftime(10, 5), os.tmpname():match("^/tmp/") ~= nil)
local name = os.tmpname() local f = assert(io.open(name, "w")) print(f:write("a\\n", 1, " ", 2.5, "\\nlast"), io.type(f)) f:close() print(io.type(f), io.type(42))
for l in io.lines(name) do io.write("[", l, "]") end print() for l in io.lines(name, "L") do io.write(#l, " ") end print()
f = io.open(name) print(f:read("n", "n", "l", "a")) print(f:seek("cur"), f:seek("set", 1), f:read(2), f:seek("end"), f:read(1), f:read("a"), f:read("l")) f:close()
f = io.open(name, "a+") f:write("\\nappended") f:seek("set") print(#f:read("a")) f:close() f = io.open(name, "rb") print(f:read(0), f:read(1), #f:read("a")) f:close()
print(io.open("/no/such/file"), pcall(io.lines, "/no/such/file"), os.remove(name), os.remove(name), os.rename("/no/a", "/no/b"))
io.write("no newline") io.write("\\n") io.stdout:write("stdout ", 1, "\\n") print(io.output() == io.stdout, io.input() == io.stdin, io.stdout:seek("cur") == nil or true)
local tmp = os.tmpname() local o = io.output(tmp) io.write("redirected") io.close() io.output(io.stdout) print(io.open(tmp):read("a")) os.remove(tmp)
print(os.execute() , os.execute("exit 3"), os.execute("true"), io.popen("echo piped"):read("l"), pcall(io.popen, "echo x", "z"))
print(pcall(f.read, f), tostring(f), tostring(io.stdout):match("^file %("))
"""),
("utf8_library", """
print(utf8.char(72, 228, 8364, 128512), utf8.charpattern, utf8.len("häl€😀"), utf8.len("\\xff"), utf8.len("abc", 4), utf8.len("abc", 5 - 1))
for p, c in utf8.codes("aä€😀") do io.write(p, "=", c, " ") end print()
print(utf8.codepoint("häl", 1, -1)) print(utf8.offset("häl€", 3), utf8.offset("häl€", -1), utf8.offset("abc", 5), pcall(utf8.codepoint, "\\xff"))
print(pcall(utf8.char, -1), utf8.char(0x7FFFFFFF):byte(1, -1)) print(utf8.len("\\u{D7FF}"), utf8.len("\\u{10FFFF}"), utf8.len("\\u{110000}", 1, -1, true), utf8.len("\\u{110000}"))
for _, c in utf8.codes("ab", true) do io.write(c, " ") end print() print(pcall(function() for _ in utf8.codes("a\\xffb") do end end))
print(("héllo"):upper(), ("ÀB"):lower(), #("€"):rep(3), ("日本語"):len(), utf8.len("日本語"), ("日本語"):sub(1, 3), ("日本語"):match(utf8.charpattern))
local t = {} for ch in ("a€b"):gmatch(utf8.charpattern) do t[#t + 1] = #ch end print(table.concat(t, ","))
"""),
("math_library", """
print(math.abs(-5), math.abs(math.mininteger), math.ceil(1.2), math.ceil(-1.2), math.floor(1.8), math.floor(-1.8), math.floor(2^70), math.max(1, 5, 3), math.min(4.5, 2), pcall(math.max))
print(math.sqrt(16), math.sqrt(2), math.sin(0), math.cos(0), math.tan(0), math.asin(1), math.acos(1), math.atan(1), math.atan(1, -1), math.exp(0), math.log(1), math.log(8, 2), math.log(100, 10), math.log(2, 4))
print(math.pi, math.huge, -math.huge, math.maxinteger, math.mininteger, math.tointeger(3.0), math.tointeger(3.5), math.tointeger("x"), math.type(1), math.type(1.0), math.type("1"), math.ult(1, -1), math.ult(-1, 1))
print(math.fmod(7, 3), math.fmod(-7, 3), math.fmod(7, -3), math.fmod(7.5, 2), pcall(math.fmod, 1, 0), math.fmod(1, 0.0), math.modf(3.7), math.modf(-3.7), math.modf(math.huge), math.modf(5))
math.randomseed(42) local a = {math.random(), math.random(10), math.random(5, 7)} math.randomseed(42) local b = {math.random(), math.random(10), math.random(5, 7)} print(a[1] == b[1], a[2] == b[2], a[3] == b[3], a[1] >= 0 and a[1] < 1, a[2] >= 1 and a[2] <= 10)
print(pcall(math.random, 0), pcall(math.random, 2, 1), math.random(3, 3), pcall(math.random, 1, 2, 3), math.random(math.mininteger, math.maxinteger) ~= nil)
print(7 // 2, 7.0 // 2, -7 // 2, 7 % 3, -7 % 3, 7 % -3, 2^0.5 == math.sqrt(2), 10 // 3 * 3 + 10 % 3, 5 // 0.0, -5 // 0.0, 0.0 / 0.0 ~= 0.0 / 0.0, 3 % math.huge, -3 % math.huge, pcall(function() return 3 % 0 end))
local s = 0 for i = 1, 1000 do s = s + math.sin(i) * math.cos(i) end print(string.format("%.10f", s))
"""),
("bitwise_ops", """
print(5 & 3, 5 | 3, 5 ~ 3, ~5, 1 << 4, 256 >> 4, -1 >> 1, -1 << 1, 1 << 63, 1 << 64, 1 >> -1, 1 << -1, 2^53 | 0, pcall(function() return "3" | 0 end), pcall(function() return 1.5 | 0 end), pcall(function() return "a" | 0 end))
print(0xFF, 0xff & 0x0f, 0x7fffffffffffffff, 0xffffffffffffffff, 0x1p4, 0x.8, 0xA.8p0, 1e2 | 0, math.maxinteger // 1 | 0, -0x8000000000000000)
local function popcount(n) local c = 0 while n ~= 0 do c = c + (n & 1) n = n >> 1 end return c end print(popcount(255), popcount(-1), popcount(0x5555555555555555))
local h = 5381 for i = 1, 200 do h = ((h << 5) + h + i) & 0xFFFFFFFF end print(h)
local x = 0x12345678 print(string.format("%08x %08x %08x", x << 4 & 0xFFFFFFFF, x >> 4, ((x << 8) | (x >> 24)) & 0xFFFFFFFF))
print(3 & 2.0, pcall(function() return 3 & 2.5 end), pcall(function() return 2^64 | 0 end), math.tointeger(2^63), -2^63 | 0)
"""),
("sorting_algorithms", """
local function qsort(t, lo, hi) lo, hi = lo or 1, hi or #t if lo < hi then local p, i = t[hi], lo - 1 for j = lo, hi - 1 do if t[j] <= p then i = i + 1 t[i], t[j] = t[j], t[i] end end t[i + 1], t[hi] = t[hi], t[i + 1] qsort(t, lo, i) qsort(t, i + 2, hi) end return t end
local data = {} local x = 12345 for i = 1, 300 do x = (x * 1103515245 + 12345) % 2147483648 data[i] = x % 1000 end
local a, b = {table.unpack(data)}, {table.unpack(data)} qsort(a) table.sort(b) local same = true for i = 1, #a do same = same and a[i] == b[i] end print(same, a[1], a[300], a[150])
local function msort(t) if #t <= 1 then return t end local m = #t // 2 local l, r = msort({table.unpack(t, 1, m)}), msort({table.unpack(t, m + 1)}) local o, i, j = {}, 1, 1 while i <= #l and j <= #r do if l[i] <= r[j] then o[#o + 1] = l[i] i = i + 1 else o[#o + 1] = r[j] j = j + 1 end end for k = i, #l do o[#o + 1] = l[k] end for k = j, #r do o[#o + 1] = r[k] end return o end
local c = msort(data) local ok = true for i = 1, #c do ok = ok and c[i] == b[i] end print(ok)
local words = {"banana", "apple", "Cherry", "date", "Elder", "fig"} table.sort(words, function(p, q) return p:lower() < q:lower() end) print(table.concat(words, " "))
local recs = {} for i = 1, 20 do recs[i] = {k = i % 3, i = i} end table.sort(recs, function(p, q) if p.k ~= q.k then return p.k < q.k end return p.i < q.i end) local out = {} for _, r in ipairs(recs) do out[#out + 1] = r.k .. ":" .. r.i end print(table.concat(out, " "))
print(pcall(table.sort, {3, 1, 2}, function(p, q) return true end))
"""),
("string_buffer_programs", """
local buf = {} for i = 1, 1000 do buf[#buf + 1] = string.format("%03d", i % 1000) end local s = table.concat(buf) print(#s, s:sub(1, 12), s:sub(-6), select(2, s:gsub("00", "")))
local words = {} for w in ("the quick brown fox jumps over the lazy dog the end"):gmatch("%a+") do words[w] = (words[w] or 0) + 1 end local keys = {} for k in pairs(words) do keys[#keys + 1] = k end table.sort(keys, function(p, q) if words[p] ~= words[q] then return words[p] > words[q] end return p < q end) for i = 1, 3 do io.write(keys[i], "=", words[keys[i]], " ") end print()
local function split(str, sep) local out = {} for piece in (str .. sep):gmatch("(.-)" .. sep:gsub("%p", "%%%0")) do out[#out + 1] = piece end return out end print(table.concat(split("a,b,,c", ","), "|"), #split("", ","), table.concat(split("1.2.3", "."), "+"))
local function trim(s) return (s:gsub("^%s+", ""):gsub("%s+$", "")) end print("[" .. trim("  hi there \\t\\n") .. "]")
local csv = "name,age\\nann,30\\nbob,25" local rows = {} for line in csv:gmatch("[^\\n]+") do local r = {} for f in line:gmatch("[^,]+") do r[#r + 1] = f end rows[#rows + 1] = r end print(#rows, rows[2][1], rows[3][2], tonumber(rows[2][2]) + tonumber(rows[3][2]))
local function b64(data) local b = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/" return ((data:gsub(".", function(x) local r, byte = "", x:byte() for i = 8, 1, -1 do r = r .. (byte % 2 ^ i - byte % 2 ^ (i - 1) > 0 and "1" or "0") end return r end) .. "0000"):gsub("%d%d%d?%d?%d?%d?", function(x) if #x < 6 then return "" end local c = 0 for i = 1, 6 do c = c + (x:sub(i, i) == "1" and 2 ^ (6 - i) or 0) end return b:sub(c + 1, c + 1) end) .. ({"", "==", "="})[#data % 3 + 1]) end print(b64("Hello, World!"), b64("ab"), b64(""))
"""),
("numeric_algorithms", """
local function sieve(n) local p, out = {}, {} for i = 2, n do p[i] = true end for i = 2, n do if p[i] then out[#out + 1] = i for j = i * i, n, i do p[j] = false end end end return out end local pr = sieve(10000) print(#pr, pr[#pr], pr[100])
local function gcd(a, b) while b ~= 0 do a, b = b, a % b end return a end print(gcd(1071, 462), gcd(2^40 // 1, 6^10 // 1), gcd(17, 5))
local function fib(n) local a, b = 0, 1 for _ = 1, n do a, b = b, a + b end return a end print(fib(50), fib(90), fib(92), fib(93))
local function fact(n) return n <= 1 and 1 or n * fact(n - 1) end print(fact(20), fact(21), fact(25.0))
local function pow(b, e, m) local r = 1 b = b % m while e > 0 do if e & 1 == 1 then r = r * b % m end b = b * b % m e = e >> 1 end return r end print(pow(2, 100, 1000007), pow(3, 200, 13), pow(7, 1 << 40, 998244353))
local function isqrt(n) local x = n while true do local y = (x + n // x) // 2 if y >= x then return x end x = y end end print(isqrt(10^12 // 1), isqrt(99), isqrt(2^62 // 1))
local s = 0 for i = 1, 100000 do s = s + (i % 7) * (i % 11) end print(s) local f = 0.0 for i = 1, 1000 do f = f + 1 / i end print(string.format("%.12f", f))
local m = {} for i = 1, 20 do m[i] = {} for j = 1, 20 do m[i][j] = (i * j) % 10 end end local tr, det = 0, 0 for i = 1, 20 do tr = tr + m[i][i] end print(tr)
print(string.format("%.15g", math.sqrt(2)), string.format("%.15g", math.pi), 7 / 3, 1e15 / 3, 2^-1074, 2^1023 * 2 == math.huge)
"""),
("recursion_and_stack", """
local function depth(n) if n == 0 then return 0 end return 1 + depth(n - 1) end print(depth(100), depth(5000), pcall(depth, 150000))
local function tail(n, acc) if n == 0 then return acc end return tail(n - 1, acc + 1) end print(tail(1000000, 0))
local function ack(m, n) if m == 0 then return n + 1 elseif n == 0 then return ack(m - 1, 1) else return ack(m - 1, ack(m, n - 1)) end end print(ack(2, 3), ack(3, 3))
local ok, err = pcall(function() local function inf() return 1 + inf() end return inf() end) print(ok, (tostring(err):gsub("^.-:%d+: ", "")))
local co = coroutine.wrap(function() return depth(1000) end) print(co())
local t = {} for i = 1, 100 do t = {t} end local function d(x) return x[1] and 1 + d(x[1]) or 0 end print(d(t))
local s = string.rep("(", 150) .. "1" .. string.rep(")", 150) print(load("return " .. s) ~= nil, load("return " .. string.rep("(", 300) .. "1" .. string.rep(")", 300)))
print(select("#", table.unpack({}, 1, 200)), pcall(table.unpack, {}, 1, 1e7), pcall(string.rep, "x", 1 << 40))
"""),
("integer_for_and_overflow", """
for i = math.maxinteger - 1, math.maxinteger do io.write(i, " ") end print()
for i = math.mininteger, math.mininteger + 1 do io.write(i, " ") end print()
for i = 1, 3 do local j = i i = i * 10 io.write(j, ":", i, " ") end print()
for i = 3, 1, -1 do io.write(i, " ") end for i = 1, 2, 0.5 do io.write(i, " ") end for i = 0.1, 0.35, 0.1 do io.write(i, " ") end print()
print(pcall(function() for i = 1, "x" do end end), pcall(function() for i = "a", 2 do end end), pcall(function() for i = 1, 2, {} do end end))
for i = 1, 3.5 do io.write(i, " ") end for i = 1.5, 3 do io.write(i, " ") end for i = 3, 1.5, -1 do io.write(i, " ") end print()
for i = math.maxinteger, math.maxinteger + 1.0 do io.write("float-limit ", i, " ") break end print()
print(math.maxinteger + 0.0 == math.maxinteger, math.maxinteger + 0.0, 2^63 == math.maxinteger + 1.0, math.mininteger + 0.0 == math.mininteger, 1 < 1.0000000000000002, math.maxinteger < math.huge, 9007199254740993 < 9007199254740992.0)
local n = 0 for i = 1, 1e6 do n = n + 1 end print(n, 1e6 == 10^6, math.type(1e6))
"""),
("table_edge_cases", """
local t = {} t[1] = "a" t[2] = "b" t[4] = "d" print(#t == 2 or #t == 4, t[3]) t[3] = "c" print(#t)
local u = {n = 1, [1] = "x", [2.0] = "two", ["3"] = "three"} print(u[2], u[3], u["3"], #u, u[1.0]) u[2^53] = "big" print(u[2^53], u[9007199254740992])
local nan = 0 / 0 print(pcall(function() local t = {} t[nan] = 1 end), ({})[nan], pcall(function() local t = {} t[nil] = 1 end), ({})[nil])
local keys = {} for k in pairs({10, 20, 30, a = 1}) do keys[#keys + 1] = tostring(k) end table.sort(keys) print(table.concat(keys, " "))
local r = {} for i = 1, 10 do r[i] = i end for i = 1, 10 do r[i] = nil end print(#r, next(r)) for i = 1, 5 do r[i] = i end print(#r)
local m = {} for i = 1, 100 do m["k" .. i] = i end local c = 0 for k, v in pairs(m) do c = c + 1 m[k] = nil end print(c, next(m))
local mod = {} for i = 1, 20 do mod[i] = i end for k, v in pairs(mod) do if v % 2 == 0 then mod[k] = nil end end local left = 0 for _ in pairs(mod) do left = left + 1 end print(left)
print(pcall(next, {}, "nokey"), #"", #{{}, {}}, #{nil}, #{nil, 2} >= 0, select("#", table.unpack({nil, nil}, 1, 2)))
local nest = {a = {b = {c = {d = "deep"}}}} print(nest.a.b.c.d, nest["a"]["b"].c["d"], pcall(function() return nest.x.y end))
local big = {} for i = 1, 100000 do big[i] = i end print(#big, big[#big]) for i = 1, 99990 do big[#big] = nil end print(#big)
local arr = {1, 2, 3} arr[#arr + 1] = 4 arr[#arr + 1] = 5 print(#arr, table.concat(arr, ","), ipairs(arr) ~= nil)
"""),
("string_coercions_comparisons", """
print("10" + 5, "3" * "4", 10 .. 20, "0x10" + 0, "1e1" + 0, " 5 " + 0, pcall(function() return "abc" + 1 end), pcall(function() return {} .. "x" end))
print("a" < "b", "a" < "B", "" < "a", "abc" < "abd", "Z" < "a", "10" < "9", pcall(function() return 1 < "2" end), "a\\0b" < "a\\0c", "a\\0" > "a")
print(1 == 1.0, "1" == 1, 0 == -0, 1/0 == math.huge, "abc" == "abc", {} == {}, math.type(2^31 | 0), 2^31 | 0, "10" + 0 == 10, "0x10" == 16)
print(tostring(1e100), tostring(-1e-100), tostring(1/3), tostring(100), tostring(100.0), tostring(1e15), tostring(2^63), tostring(-2^63), 123456.789e3, 0.1 + 0.2 == 0.3, 255 // 1 | 0)
print(#tostring(print) > 0, tostring(print):match("^builtin") or tostring(print):match("^function"), tostring({}):match("^table: ") ~= nil, tostring(coroutine.create(print)):match("^thread: ") ~= nil)
print(("x"):rep(3) == "xxx", "a" .. "b" .. "c" == "abc", "a" .. 1 + 2, 2 ^ 3 ^ 2, -2 ^ 2, not nil == true, not 0, 1 and 2, nil and 1, false or "d", nil or false, 1 or error("no"))
print(#"abc" + 1, -"2", - -"2", "5" // 2, "5" % 3, 10 // "3", pcall(function() return -{} end), 2^"2", "2"^2, pcall(function() return "8" | 1 end), pcall(function() return "8.5" | 1 end))
"""),
]
