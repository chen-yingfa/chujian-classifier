from pathlib import Path
import json
from collections import defaultdict


def dump_json(data, file):
    json.dump(
        data,
        open(file, 'w', encoding='utf8'),
        ensure_ascii=False,
        indent=4,
    )


glyph_count_file = Path('glyph_count.json')
if not glyph_count_file.exists():
    data_dir = Path('chujian/glyphs')
    glyph_cnt = {}
    for glyph_dir in sorted(data_dir.iterdir()):
        glyph_cnt[glyph_dir.name] = len(
            [image for image in glyph_dir.iterdir()])
    dump_json(glyph_cnt, 'glyph_count.json')
else:
    glyph_cnt = json.load(open(glyph_count_file, 'r', encoding='utf-8'))


merged = defaultdict(int)
for glyph, cnt in glyph_cnt.items():
    # Discard all glyphs containing these chars.
    skip_chars = [
        '?'
        '□', '■',
        '○', '●',
        '△', '▲',
        '☆', '★',
        '◇', '◆',
    ]
    if any(c in glyph for c in skip_chars):
        continue

    # Normalize the glyph label
    rm_chars = [
        '=', 'None'
    ]
    for c in rm_chars:
        glyph = glyph.replace(c, '')

    # Replace brackets
    for c in ['（', '〈', '[']:
        glyph = glyph.replace(c, '(')
    for c in ['）', '〉', ']']:
        glyph = glyph.replace(c, ')')
    # If the label ends with (.), use the glyph inside the brackets
    if glyph == '':
        continue

    if glyph[-1] == ')':
        for i in range(len(glyph) - 1, -1, -1):
            if glyph[i] == '(':
                # "（*）"
                if i == 0:
                    glyph = glyph[1:-1]
                # "a（*）"
                if i == 1:
                    glyph = glyph[0]
                # "*}（*）"
                elif glyph[i-1] == '}':
                    glyph = glyph[i+1:-1]
                break
    # "A→B"
    if '→' in glyph:
        glyph = glyph.split('→')[1]

    merged[glyph] += cnt


# Remove the glyphs with less than 10 samples
merged = {k: v for k, v in merged.items() if v >= 10}
merged_sorted = sorted(merged.items(), key=lambda x: x[1], reverse=True)
merged = {k: v for k, v in merged_sorted}
dump_json(merged_sorted, 'merged_glyph_count_sorted.json')

dump_json(merged, 'merged_glyph_count.json')

print(f'Found {len(merged)} glyphs')
