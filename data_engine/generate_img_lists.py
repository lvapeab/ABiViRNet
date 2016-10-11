import glob

base_path = '/media/HDD_2TB/DATASETS/MSVD/'

# Inputs
split_lists = ['train_list.txt', 'val_list.txt', 'test_list.txt']
imgs_format = '.jpg'
path_imgs = 'Images'
path_files = 'Annotations'

# Outputs
out_lists = ['train_imgs_list.txt', 'val_imgs_list.txt', 'test_imgs_list.txt']
counts_lists = ['train_imgs_counts.txt', 'val_imgs_counts.txt', 'test_imgs_counts.txt']

# Code
print 'Listing all images from all videos...'

len_base = len(base_path)
for s, o, c in zip(split_lists, out_lists, counts_lists):
    s = open(base_path+'/'+path_files+'/'+s, 'r')
    o = open(base_path+'/'+path_files+'/'+o, 'w')
    c = open(base_path+'/'+path_files+'/'+c, 'w')
    for line in s:
        video = line.strip('\n')
        this_path = base_path+'/'+path_imgs+"/video_"+video+"/*"+imgs_format
        images = glob.glob(this_path)
        for im in images:
            #o.write(path_imgs+"/video_"+video+"/"+im+'\n') # store each image path
            o.write(im[len_base:]+'\n')
        c.write(str(len(images))+'\n') # store counts
    s.close()
    o.close()
    c.close()
    
print 'Done!'

