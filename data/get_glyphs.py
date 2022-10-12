from pathlib import Path
import json


data_dir = Path('chujian/glyphs')
glyphs = {}
for glyph_dir in sorted(data_dir.iterdir()):
    glyphs[glyph_dir.name] = len([image for image in glyph_dir.iterdir()])

json.dump(
    glyphs, 
    open('glyph_count.json', 'w', encoding='utf-8'), 
    indent=4, 
    ensure_ascii=False,
)
