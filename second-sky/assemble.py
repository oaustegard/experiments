"""Inject work/sky.json into app.html -> second-sky.html"""
import json
sky = open("work/sky.json").read()
html = open("app.html").read().replace("/*SKY*/null", sky.replace("</", "<\\/"))
open("second-sky.html", "w").write(html)
print("second-sky.html", round(len(html) / 1e6, 2), "MB")
