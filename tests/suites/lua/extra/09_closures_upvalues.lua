local function counter() local n = 0 return function() n = n + 1 return n end, function() return n end end
local inc, get = counter() inc() inc() print(get())
local fns = {} for i = 1, 3 do fns[i] = function() return i end end print(fns[1](), fns[2](), fns[3]())
local acc = {} do local x = 0 for i = 1, 3 do local y = i * 10 acc[i] = function() x = x + y return x end end end print(acc[1](), acc[2](), acc[3](), acc[1]())
local function shared() local v = 0 return function() v = v + 1 return v end, function() v = v * 10 return v end end
local a, b = shared() a() a() print(b(), a())
local function deep(n) if n == 0 then return function() return "bottom" end end local f = deep(n - 1) return function() return f() end end print(deep(50)())
local t = {} for i = 1, 3 do local j = i t[#t + 1] = function() j = j + 1 return j end end print(t[1](), t[1](), t[2](), t[3]())
print(select("#", (function(...) return ... end)(1, nil, nil)), (function(...) local a, b = ... return b end)(1, 2))
