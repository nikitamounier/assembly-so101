from pathlib import Path
import sys,shutil
p=Path(sys.argv[1]);src=Path(__file__).parent
shutil.copy2(src/'continuation.h',p/'continuation.h')
f=p/'so101_contact_search.c';s=f.read_text();anchor='static int contact_search('
s=s.replace(anchor,'#include "continuation.h"\n'+anchor,1)
old='collider(&view, d, con, g1, g2, margin)';assert s.count(old)==1
s=s.replace(old,'continue_contacts(collider,&view,d,con,g1,g2,margin)');f.write_text(s)
