# Retrieves the images of a given split and sorts them according to that split
import shutil
from common import create_dir_if_not_exists

image_dir = '/data/DATASETS/Flickr8k/Images'
annotatios_dir = '/data/DATASETS/Flickr8k/Annotations'
split_name = 'val'
dest_dir = image_dir + '/' +  split_name + '_images'
ext = '.jpg'



with open(annotatios_dir + '/' + split_name + '_list_ids.txt') as f:
    lines = f.readlines()

create_dir_if_not_exists(dest_dir)
n_items = len(str(len(lines))) + 1
i = 0
for filename in lines:
    i += 1
    shutil.copyfile(image_dir + '/' + filename[:-1] + ext, dest_dir + '/' + str(i).zfill(n_items) + ext)
