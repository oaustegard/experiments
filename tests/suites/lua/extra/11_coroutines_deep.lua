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
