from pathlib import Path
from shutil import copyfile
from tqdm import tqdm


def load_tsv(file) -> list:
    return [line.rstrip().split('\t') for line in open(file)]

def get_files(map_file: Path, dst_dir: Path):
    img2label = load_tsv(map_file)
    print(img2label[:3])
    n = len(img2label)
    dst_dir.mkdir(exist_ok=True)
    for row in tqdm(img2label[1:]):
        src_file = row[0]
        label_name = row[1]
        juan_idx = row[2]
        image_num = row[3]
        idx_list = row[4]
        
        label_dir = dst_dir / label_name
        label_dir.mkdir(exist_ok=True)
        dst_file = label_dir / src_file.split('/')[-1]
        copyfile(src_file, dst_file)
    


qinghuaba_map_file = Path("image2label_details_qinghuaba.csv")
map_file = Path("image2label_details.csv")

if __name__ == '__main__':
    get_files(map_file, Path('1-7'))
    # get_files(qinghuaba_map_file, Path('8'))
