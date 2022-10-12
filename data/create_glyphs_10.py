from pathlib import Path

src_dir = Path('./chujian/glyphs')
dst_dir = Path('./chujian/glyphs_10')
for glyph_dir in src_dir.iterdir():
    if not glyph_dir.is_dir():
        continue
    glyph_name = glyph_dir.name
    dst_glyph_dir = dst_dir / glyph_name
    dst_glyph_dir.mkdir(exist_ok=True, parents=True)
    for image in glyph_dir.iterdir():
        dst_file = dst_glyph_dir / image.name
        copyfile(image, dst_file)
        break