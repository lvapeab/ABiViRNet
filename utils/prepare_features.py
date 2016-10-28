import numpy as np
from common import create_dir_if_not_exists


###### Parameters

ROOT_PATH = '/media/HDD_2TB/DATASETS/'

base_path = ROOT_PATH +'/Flickr8k/Features/'
features = 'KCNN' # KCNN, Scenes, Objects
base_path_save = base_path + features

feats_paths = ['train_' + features + '_features.csv',
               'val_' + features + '_features.csv',
               'test_' + features + '_features.csv']

names_lists = ['train_list.txt', 'val_list.txt', 'test_list.txt']
folders_save = ['train', 'val', 'test']

apply_L2 = False
n_feats = 1024


############

if apply_L2:
    file_save = features + '_L2'
else:
    file_save = features


def csv2npy():

    # Process each data split separately
    for n, f, fs in zip(names_lists, feats_paths, folders_save):
        print "Preparing features %s" % f
        feats_dict = dict()
        # Get file names
        names = []
        with open(base_path + '/' + n, 'r') as file:
            for line in file:
                line = line.rstrip('\n')
                line = line.split('.')[0]
                names.append(line)
        # Get features
        with open(base_path + '/' + f, 'r') as file:
            for i, line in enumerate(file):
                feats = np.fromstring(line.rstrip('\n'), sep=',')
                if(apply_L2):
                    feats = feats/np.linalg.norm(feats, ord=2)
                # Insert in dictionary
                feats_dict[names[i]] = feats[:n_feats]

        # Store dict
        print "Saving features in %s" % (base_path_save + '/' + fs + '/' + file_save + '.npy')
        create_dir_if_not_exists(base_path_save +'/'+ fs)
        np.save(base_path_save + '/' + fs + '/' + file_save + '.npy', feats_dict)
        print


if __name__ == "__main__":
    csv2npy()
