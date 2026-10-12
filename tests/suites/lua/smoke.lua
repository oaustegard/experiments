-- Short deterministic smoke script run by the per-.c-file tests of the lua suite.
-- It touches every library and the main VM paths; its stdout is compared with
-- the output of a gcc-built lua.
local function show(...)
  local t = table.pack(...)
  for i = 1, t.n do t[i] = tostring(t[i]) end
  print(table.concat(t, " "))
end

-- numbers, integer/float distinction, arithmetic
show(1 + 2, 7 // 2, 7 / 2, -7 // 2, -7 % 3, 7 % -3, 2 ^ 10, 10 // 0.0, -10 % math.huge)
show(math.maxinteger, math.mininteger, math.maxinteger + 1 == math.mininteger)
show(1e15, 1e100, 0.1, 1 / 3, 123456789012, 3 | 4, 7 & 2, 5 ~ 3, ~0, 1 << 62, 1 >> 1, 1 << 64)
show(math.type(1), math.type(1.0), math.type("1"), 3 == 3.0, math.tointeger(3.0), math.tointeger(3.5))
show(math.floor(-3.5), math.ceil(-3.5), math.abs(-4), math.max(3, 9, 2), math.min(3, 9, 2), math.fmod(7, 3))
show(math.sqrt(2), math.sin(1), math.cos(1), math.exp(1), math.log(10), math.log(8, 2), math.pi)
show(string.format("%5.2f|%-5d|%05d|%x|%X|%o|%e|%g|%q", 3.14159, 42, 42, 255, 255, 8, 12345.678, 0.0001, "a\nb"))
show(string.format("%s %s %s", nil, true, {} ~= nil), string.format("%10s|%-10s|", "hi", "hi"), string.format("%c%c", 72, 105))
show(tostring(1e308 * 10), tostring(-1e308 * 10), tostring(0 / 0) == tostring(0 / 0), 2^53, 2^63, math.ult(1, -1))
show("10" + 5, "0x10" + 0, 10 .. 20, "3" * "4", tonumber("  12  "), tonumber("1e2"), tonumber("z", 36), tonumber("8", 8))

-- strings
show(("hello"):upper(), ("HeLLo"):lower(), ("abc"):rep(3, "-"), ("abc"):reverse(), ("abc"):len(), #"héllo")
show(("hello world"):find("o w"), ("hello world"):find("o", 6), ("hello"):find("l+"), ("hello"):match("(h)(e)"))
show(("key = value"):match("^(%w+)%s*=%s*(%w+)$"))
show(("a,b,,c"):gsub(",", ";"), ("hello"):gsub("l", {l = "L"}), ("abc"):gsub("%w", "%0%0"), ("abc"):gsub("", "-"))
show(("x"):byte(), string.char(65, 66, 67), ("abc"):sub(2), ("abc"):sub(-2, -1), ("abc"):sub(0), ("abc"):sub(5))
for k, v in string.gmatch("a=1, b=2, c=3", "(%w+)=(%w+)") do io.write(k, ":", v, " ") end
print()
show(string.pack("<i4", 100):byte(1, -1))
show(string.unpack("<i4", string.pack("<i4", -2)))
show(string.packsize("i4i8"), #string.pack("z", "abc"), string.unpack("s1", string.pack("s1", "xyz")))
show(utf8.char(72, 228, 8364, 128512), utf8.len("häl€"), utf8.codepoint("€", 1), #utf8.char(128512))
for p, c in utf8.codes("aä€") do io.write(p, "=", c, " ") end
print()

-- tables, sorting, iteration
local t = {5, 2, 8, 1, 9, 3}
table.sort(t)
show(table.concat(t, ","), #t, select("#", table.unpack(t)), select(-1, table.unpack(t)))
table.sort(t, function(a, b) return a > b end)
show(table.concat(t, ","))
table.insert(t, 7); table.insert(t, 1, 0); show(table.concat(t, ","), table.remove(t), table.remove(t, 1), #t)
show(table.concat(table.move({1, 2, 3}, 1, 3, 2), ","), table.unpack({1, 2, 3}, 2))
local keys = {}
for k in pairs({a = 1, b = 2, c = 3, 10, 20}) do keys[#keys + 1] = tostring(k) end
table.sort(keys); show(table.concat(keys, " "))
local big = {}
for i = 1, 1000 do big[i] = i * i end
show(#big, big[1000], next({}), rawlen({1, 2}), rawequal(big, big), rawget(big, 3))

-- closures, varargs, multiple returns, goto
local function counter()
  local n = 0
  return function() n = n + 1; return n end
end
local c1, c2 = counter(), counter()
c1(); c1()
show(c1(), c2())
local function va(...) return select("#", ...), ... end
show(va(), va(nil), va(1, nil, nil))
local function fib(n) if n < 2 then return n end return fib(n - 1) + fib(n - 2) end
show(fib(20))
local function tail(n, acc) if n == 0 then return acc end return tail(n - 1, acc + n) end
show(tail(100000, 0))
do
  local i = 0
  ::top::
  i = i + 1
  if i < 5 then goto top end
  show("goto", i)
end
for i = 10, 1, -3 do io.write(i, " ") end
for i = 1.0, 2.0, 0.5 do io.write(i, " ") end
print()
local a <const> = 5
do
  local x <close> = setmetatable({}, {__close = function() show("closed") end})
  show("in scope", a)
end

-- metatables, OOP, inheritance
local V = {}
V.__index = V
V.__add = function(a, b) return setmetatable({x = a.x + b.x}, V) end
V.__eq = function(a, b) return a.x == b.x end
V.__lt = function(a, b) return a.x < b.x end
V.__le = function(a, b) return a.x <= b.x end
V.__tostring = function(a) return "V(" .. a.x .. ")" end
V.__len = function(a) return a.x end
V.__call = function(self, y) return self.x + y end
V.__concat = function(a, b) return tostring(a) .. "|" .. tostring(b) end
V.__unm = function(a) return setmetatable({x = -a.x}, V) end
function V.new(x) return setmetatable({x = x}, V) end
local v1, v2 = V.new(1), V.new(2)
show(tostring(v1 + v2), v1 == V.new(1), v1 < v2, v1 <= v2, #v2, v2(10), v1 .. v2, tostring(-v1))
local prox = setmetatable({}, {__index = function(_, k) return k .. "!" end, __newindex = function(t, k, v) rawset(t, k, v * 2) end})
prox.z = 21
show(prox.foo, prox.z, getmetatable("").__index == string)
local weak = setmetatable({}, {__mode = "k"})
weak[{}] = 1
collectgarbage()
show(next(weak), collectgarbage("count") > 0)

-- errors, pcall, xpcall, error objects
show(pcall(error, "boom"))
show(pcall(error, {code = 1}) == false)
show(pcall(function() local x = nil; return x.y end))
show(pcall(function() return 1 + {} end))
show(pcall(function() return #5 end))
show(select(2, pcall(error, "lvl2", 2)))
show(xpcall(function() error("E") end, function(m) return "handled: " .. m end))
show(pcall(string.rep, "x", -1), pcall(setmetatable, 1, {}))
show(tostring(nil), tostring(true), type(print), type(nil), type({}), type("x"), type(2), type(coroutine.create(print)))
show(load("return 1 + 1")(), load("syntax error here"), loadstring)
show(assert(1 == 1, "no"), pcall(assert, false, "assertion msg"), pcall(assert, nil))

-- coroutines
local co = coroutine.create(function(a, b)
  local c = coroutine.yield(a + b)
  local d, e = coroutine.yield(c * 2)
  return d + e, "done"
end)
show(coroutine.resume(co, 1, 2))
show(coroutine.resume(co, 10))
show(coroutine.resume(co, 3, 4))
show(coroutine.resume(co), coroutine.status(co))
local gen = coroutine.wrap(function() for i = 1, 3 do coroutine.yield(i) end end)
show(gen(), gen(), gen())
show(coroutine.isyieldable(), select(2, coroutine.running()))
local co2 = coroutine.wrap(function() error("in co") end)
show(pcall(co2))

-- os / io / debug basics
show(type(os.time()), os.time({year = 2000, month = 1, day = 1, hour = 12}) ~= nil, os.date("!%Y-%m-%d %H:%M:%S", 86400))
show(os.getenv("HCC_SURELY_UNSET_VARIABLE"), type(os.clock()), os.date("!%A %B", 0))
local name = os.tmpname()
local f = assert(io.open(name, "w"))
f:write("line1\n", 2, "\n", 3.5, "\nlast")
f:close()
for l in io.lines(name) do io.write("[", l, "]") end
print()
f = io.open(name)
show(f:read("l"), f:read("n"), f:read("n"), f:read("a"), f:read("a"), f:read("l"))
show(f:seek("set", 2), f:read(3), f:seek("cur"), f:seek("end"))
f:close()
show(io.type(f), io.type(io.stdout), io.type(42), pcall(f.read, f))
os.remove(name)
show(io.open("/nonexistent/dir/file"))
local info = debug.getinfo(1, "Sl")
show(info.short_src ~= nil, info.currentline, debug.traceback("tb", 1):sub(1, 12))
show(#arg, arg[0] ~= nil, _VERSION)
io.stdout:write("done\n")
