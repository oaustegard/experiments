CREATE VIRTUAL TABLE gp USING geopoly(name);
INSERT INTO gp(_shape, name) VALUES('[[0,0],[10,0],[10,10],[0,10],[0,0]]', 'square'),('[[5,5],[15,5],[10,15],[5,5]]', 'triangle'),('[[20,20],[30,20],[30,30],[20,30]]', 'far');
SELECT name, geopoly_area(_shape) FROM gp ORDER BY name;
SELECT name FROM gp WHERE geopoly_overlap(_shape, '[[8,8],[12,8],[12,12],[8,12]]') ORDER BY name;
SELECT name FROM gp WHERE geopoly_within(_shape, '[[-1,-1],[11,-1],[11,11],[-1,11]]') ORDER BY name;
SELECT name FROM gp WHERE geopoly_contains_point(_shape, 7, 7) ORDER BY name;
SELECT geopoly_json('[[0,0],[1,0],[1,1]]'), geopoly_bbox('[[0,0],[3,0],[3,4],[0,4]]') IS NOT NULL, geopoly_ccw('[[0,0],[0,1],[1,1],[1,0]]') IS NOT NULL;
SELECT geopoly_regular(0,0,5,6) IS NOT NULL, round(geopoly_area(geopoly_regular(0,0,5,6)), 4), geopoly_svg('[[0,0],[1,0],[1,1]]') IS NOT NULL;
SELECT geopoly_xform('[[0,0],[1,0],[1,1]]', 1, 0, 0, 1, 5, 5), geopoly_group_bbox(_shape) IS NOT NULL FROM gp;
