import glob
import os
import numpy as np

base_path = '/media/HDD_2TB/DATASETS/MSVD/'
features_path = 'Features/Full_Features'
output_path = 'Features'

n_frames_per_video_subsample = 26  # subsample fixed number of equidistant frames per video
repeat_frames = False   # decides if we are going to repeate some frames when needed for filling the desired
                        # "n_frames_per_video_subsample", or we are simply filling the video frames with 0s


# Inputs
features_name = 'C3D_fc8_ImageNet'
features_files = ['train_' + features_name + '_features.csv', 'val_' + features_name + '_features.csv', 'test_' + features_name + '_features.csv']
features_counts_files = ['train_' + features_name + '_counts.txt', 'val_' + features_name + '_counts.txt', 'test_' + features_name + '_counts.txt']

# Outputs
out_features_name = 'C3D_fc8_ImageNet'
out_features = ['train_' + out_features_name + '.csv', 'val_' + out_features_name + '.csv', 'test_' + out_features_name + '.csv']
out_features_counts = ['train_' + out_features_name + '_counts.txt', 'val_' + out_features_name + '_counts.txt', 'test_' + out_features_name + '_counts.txt']

#########

for ff_, fc_, of_, oc_ in zip(features_files, features_counts_files, out_features, out_features_counts):
    
    print 'Processing file', base_path+'/' + features_path + '/' + ff_
    
    # Open files
    ff = open(base_path +'/' + features_path + '/' + ff_, 'r')
    fc = open(base_path +'/' + features_path + '/' + fc_, 'r')
    of = open(base_path +'/' + output_path + '/' + of_, 'w')
    oc = open(base_path +'/' + output_path + '/' + oc_, 'w')
    
    # Process each video
    for count_videos, count in enumerate(fc):
        # Calculate chosen frames
        count = int(count.strip('\n'))
        #pick_pos = np.round(np.linspace(0,count-1,n_frames_per_video_subsample)).astype('int64')
        pick_pos = np.linspace(0, count - 1, n_frames_per_video_subsample).astype('int64')
        if not repeat_frames:
            pick_pos = np.unique(pick_pos)
            count_pick = len(pick_pos)
        
        # Get all frames from current video
        feats = [[] for i in range(count)]
        for i in range(count):
            feats[i] = ff.next()
        
        # Get chosen frames
        for p in pick_pos:
            of.write(feats[p])
            oc.write(str(count_pick)+'\n')
            if count_pick != n_frames_per_video_subsample:
                print "different", count_videos
                print "num", count_pick

    ff.close()
    fc.close()
    of.close()
    oc.close()
    
    print 'Output stored in', base_path+'/' + output_path + '/' + of_
        
        
