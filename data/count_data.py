from pathlib import Path
import matplotlib.pyplot as plt

# data_dir = Path('./chujian/glyphs_small_train')
data_dir = Path('./chujian/glyphs_955/train')
# data_dir = Path('./chujian/glyphs')
glyph_cnts = {}
for glyph_dir in data_dir.iterdir():
    glyph_cnts[glyph_dir.name] = len(list(glyph_dir.iterdir()))

cnts = list(glyph_cnts.items())
cnts = sorted(cnts, key=lambda x: x[1], reverse=True)
num_samples = sum([cnt for _, cnt in cnts])
num_class = len(cnts)
print('# class:', num_class)
print('# samples:', num_samples)
for i in [2, 5, 10]:
    cnt = sum([1 for x in cnts if x[1] < i])
    proportion = round(100 * cnt / num_class, 2)
    print(f'# class with cnt < {i}: {cnt} ({proportion}%)')

print('------ Top 10 classes:')
for i in range(10):
    print(cnts[i][0], cnts[i][1])
print('------')
print('------ Bottom 10 classes:')
for i in range(1, 11):
    print(cnts[-i][0], cnts[-i][1])
print('------')


graph_file = 'class_occurrence.png'
print(f'Plotting and saving graph to {graph_file}')
cnts = [cnt for _, cnt in cnts]
plt.rc('font', size=13)
plt.plot(cnts)
plt.xlabel('Glyphs')
plt.ylabel('Number of images')
plt.yscale('log')
plt.savefig(graph_file)
