.help .tables
.print hello
.print "two words" 'and more'
.echo on
SELECT 1;
.echo off
.nosuchcommand
.width 5 10
.mode column
.headers on
SELECT 'ab', 'cd';
.timer off
.parameter set @x 5
SELECT @x + 1;
.bail off
SELECT 'end';
