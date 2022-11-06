# Data

Contains code for data processing and small example data for convenience.

## Processing Pipeline

1. `preprocess_data.py`: Reorder crawled raw data into a directory tree where subfolders are glyphs and files are images of that glyph. Each glyph is simply the label in the crawled data, which contains many weird strings. 
2. `merge_glyphs.py`: Merge different labels (8000+ -> 955) and save to another dir.
    1. Merge labels with same meaning (e.g. "A(X)" and "A(Y)" -> "A")
    2. Discard all glyphs with less than $k$ images (we use 3 or 10).
    3. Dump the data as JSON file to `DST_DIR` (a variable in the code). The most important is `DST_DIR / "glyph_to_files.json"`, which stores the each (merged) glyph and its corresponding image files (dumped by `preprocess_data.py` in step 1).
    4. `get_data_splits.py`: Split in train, dev and test by 18:1:1 ratio, and dump the file paths as JSON files.


