import glob
import os
import numpy as np

base_path = '/media/HDD_2TB/DATASETS/MSVD/'
path_features = 'Features'
path_annotations = 'Annotations'

# Inputs
features_name = 'ImageNet'

###### Files with fixed number of frames per video
# features_files = ['train_' + features_name + '.csv', 'val_' + features_name + '.csv', 'test_' + features_name + '.csv']
# features_counts = ['train_' + features_name + '_counts.txt', 'val_' + features_name + '_counts.txt', 'test_' + features_name + '_counts.txt']

###### Files all original frames of videos
features_files = ['train_' + features_name + '_all_frames.csv',
                  'val_' + features_name + '_all_frames.csv',
                  'test_' + features_name + '_all_frames.csv']
features_counts = ['train_' + features_name + '_all_frames_counts.txt',
                   'val_' + features_name + '_all_frames_counts.txt',
                   'test_' + features_name + '_all_frames_counts.txt']

# features_name = 'C3D_fc8_ImageNet'
# features_files = ['train_' + features_name + '.csv', 'val_' + features_name + '.csv', 'test_' + features_name + '.csv']
# features_counts = ['train_' + features_name + '_counts.txt', 'val_' + features_name + '_counts.txt', 'test_' + features_name + '_counts.txt']

# Outputs
out_lists = ['train_feat_list.txt', 'val_feat_list.txt', 'test_feat_list.txt']
counts_lists = ['train_feat_counts.txt', 'val_feat_counts.txt', 'test_feat_counts.txt']


#########

if not os.path.isdir(base_path+'/'+path_features+'/'+features_name):
    os.makedirs(base_path+'/'+path_features+'/'+features_name)

if not os.path.isdir(base_path+'/'+path_annotations+'/'+features_name):
    os.makedirs(base_path+'/'+path_annotations+'/'+features_name)

c_videos = 0
for f, fc, o, c in zip(features_files, features_counts, out_lists, counts_lists):
    print "Processing "+ f
    
    f = open(base_path+'/'+path_features+'/'+f, 'r')
    fc = open(base_path + '/' + path_features + '/' + fc, 'r')
    o = open(base_path+'/'+path_annotations+'/'+features_name+'/'+o, 'w')
    c = open(base_path+'/'+path_annotations+'/'+features_name+'/'+c, 'w')

    all_counts = list()
    for line in fc:
        line = line.strip('\n')
        all_counts.append(int(line))

    c_frame = 0
    c_videos_split = 0
    # Process each line in the file
    for enum,line in enumerate(f):
        frame = line.strip('\n')
        frame = np.fromstring(frame, sep=',') # covert csv line to numpy array

        this_path = "%s/video_%0.4d" %(path_features+'/'+features_name, c_videos)
        if not os.path.isdir(base_path+this_path):
            os.makedirs(base_path+this_path)
        this_path = "%s/video_%0.4d/frame_%0.4d.npy" %(path_features+'/'+features_name, c_videos, c_frame)
        # Save array in disk
        try:
            np.save(base_path+this_path, frame)
        except:
            print 'line file',enum
            print 'file name',base_path+this_path
            print 'lenvec',len(frame)
            print 'vec', frame
            print
        # Write path to file
        o.write(this_path+'\n')

        c_frame += 1

        # a complete video was processed
        if c_frame%all_counts[c_videos_split] == 0:
            c_videos += 1
            c.write(str(all_counts[c_videos_split])+'\n') # store counts
            c_videos_split += 1
            c_frame = 0



    f.close()
    fc.close()
    o.close()
    c.close()

print 'Done!'
