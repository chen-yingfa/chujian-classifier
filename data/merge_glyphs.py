from pathlib import Path
import json
from collections import defaultdict
from shutil import copyfile


def dump_json(data, file):
    json.dump(
        data,
        open(file, 'w', encoding='utf8'),
        ensure_ascii=False,
        indent=4,
    )


GLYPH_CNT_FILE = Path('glyph_count.json')
SRC_DIR = Path('/data/private/chenyingfa/chujian/glyphs')
DST_DIR = Path('/data/private/chenyingfa/chujian/glyphs_955/all')

merged_to_orig_file = DST_DIR / 'new_to_orig_name.json'


if not GLYPH_CNT_FILE.exists():
    glyph_cnt = {}
    for glyph_dir in sorted(SRC_DIR.iterdir()):
        glyph_cnt[glyph_dir.name] = len(
            [image for image in glyph_dir.iterdir()])
    dump_json(glyph_cnt, 'glyph_count.json')
else:
    glyph_cnt = json.load(open(GLYPH_CNT_FILE, 'r', encoding='utf-8'))


merged = defaultdict(int)
# Map new glyph name to old glyph name
new_to_old_name = defaultdict(list)
for glyph, cnt in glyph_cnt.items():
    orig = glyph
    # Discard all glyphs containing these chars.
    DISCARD_CHARS = [
        '?'
        '□', '■',
        '○', '●',
        '△', '▲',
        '☆', '★',
        '◇', '◆',
        '□'
    ]
    if any(c in glyph for c in DISCARD_CHARS):
        continue

    # Normalize the glyph label
    RM_STRS = [
        '=', 'None'
    ]
    for c in RM_STRS:
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
                else:
                    # "*}（*）"
                    if glyph[i-1] == '}':
                        glyph = glyph[i+1:-1]
                    # "A（*）" -> "A"
                    else:
                        glyph = glyph[0]
                break
    # "A→B"
    if '→' in glyph:
        glyph = glyph.split('→')[1]
    if glyph == '𬨭':
        glyph = '將'
    if glyph == '𫵖':
        glyph = '尸示'

    merged[glyph] += cnt
    new_to_old_name[glyph].append(orig)

# Remove the glyphs with less than 10 samples
merged = {k: v for k, v in merged.items() if v >= 10}
merged_sorted = sorted(merged.items(), key=lambda x: x[1], reverse=True)
merged = {k: v for k, v in merged_sorted}
dump_json(merged, 'merged_glyph_count.json')
new_to_old_name = {k: v for k, v in new_to_old_name.items() if k in merged}

dump_json(new_to_old_name, merged_to_orig_file)

print(f'Found {len(merged)} glyphs')

# Copy glyphs over
for i, (new_name, old_names) in enumerate(new_to_old_name.items()):
    print(f'{i} Copying {old_names} to {new_name}')
    # Create the new dir
    dst_glyph_dir = DST_DIR / new_name
    dst_glyph_dir.mkdir(exist_ok=True, parents=True)
    for old_name in old_names:
        src_dir = SRC_DIR / old_name
        for src_file in src_dir.iterdir():
            dst_file = dst_glyph_dir / src_file.name
            copyfile(src_file, dst_file)
