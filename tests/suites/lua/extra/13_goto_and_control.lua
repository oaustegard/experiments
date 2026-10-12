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
