# Data

Contains code for data processing and small example data for convenience.

## Processing Pipeline

1. `preprocess_data.py`: Reorder crawled raw data into a directory tree where subfolders are glyphs and files are images of that glyph. Each glyph is simply the label in the crawled data, which contains many weird strings. 
2. `merge_glyphs.py`: Merge different labels (8000+ -> 955) and save to another dir.
    1. Merge labels with same meaning (e.g. "A(X)" and "A(Y)" -> "A")
    2. Discard all glyphs with less than 10 examples.
3. `split_data.py`: Split into train, dev and test by 8:1:1 ratio, and save to `/some/path/train`, `/some/path/dev` and `/some/path/test`.


