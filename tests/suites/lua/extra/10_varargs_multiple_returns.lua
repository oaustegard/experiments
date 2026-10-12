local function f(...) return select("#", ...), ... end
print(f()) print(f(nil)) print(f(1, nil, 3, nil)) print((f(1, 2, 3)))
local function g() return 1, 2, 3 end
local t = {g(), g()} print(#t) t = {g(), (g())} print(#t) t = {(g())} print(#t) print(g(), 10) print(10, g()) print((g()))
local a, b, c, d = g() print(a, b, c, d) local x, y = (g()) print(x, y)
print(table.pack(g()).n, math.max(g()), string.format("%d %d %d", g()), #{n = 1, g()})
local function sum(...) local s = 0 for i = 1, select("#", ...) do s = s + (select(i, ...) or 0) end return s end print(sum(1, 2, nil, 4), sum())
local function tail(...) return ... end print(tail(), tail(nil, nil), (tail(5, 6)))
print(pcall(function(...) return ... end, 1, 2, 3))
