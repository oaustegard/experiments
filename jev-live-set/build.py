import sys
src=open('page.src.html').read(); lib=open('strudel-lib.js').read().replace('</script','<\\/script')
out=src.replace('/*__STRUDEL_LIB__*/', lib+'\n;window.StrudelLib=StrudelLib;')
open(sys.argv[1],'w').write(out); print(len(out))
