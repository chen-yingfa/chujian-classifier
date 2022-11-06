from pathlib import Path


glyphs_dir = Path('/data/private/chenyingfa/chujian/glyphs')
for glyph_dir in glyphs_dir.iterdir():
    name = glyph_dir.name
    if '（○）' in name:
        print(name, len(list(glyph_dir.iterdir())))
