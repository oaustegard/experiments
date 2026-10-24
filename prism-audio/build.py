src = open('page.src.html').read()
open('index.html', 'w').write(src.replace('/*ENGINE*/', open('engine.js').read()))
