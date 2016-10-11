# Preprocessing of MSVD dataset

The scripts stored in this folder 'data_engine' are intended to preprocess the data from the Microsoft Video Description (MSVD) dataset in order to use them as an input for building a Dataset object instance (see [staged_keras_wrapper](https://github.com/MarcBS/staged_keras_wrapper)).

Two different kinds of inputs can be used for training the video description models:

1) Raw video frames (see section 'Image lists generation')
2) Features from video frames (see section 'Image features generation')


## Folder structure

Following we describe the desired folder structure for storing the dataset-related information:

    ./Images
        video_[video_id]
            [num_image].jpg
            [num_image].jpg
    ./Annotations
        test_list.txt
        train_list.txt
        val_list.txt
        captions.id.en
    ./Features
        test_[name_feat].csv
        train_[name_feat].csv
        val_[name_feat].csv

The folder ./Images contains a set of folders 'video_[video_id]', where each folder represents a video and contains a set of frames '[num_image].jpg'.

The folder ./Annotations contains, for each set split {train, val, test}, a file with the suffix _list.txt. Containing the list of videos 'video_[video_id]' belonging to the respective split. It also contains the file 'captions.id.en', which lists all the available captions for all the videos.

The folder ./Features contains any kind of features extracted from the respective set splits (only needed if using image features instead of raw images).


## Descriptions generation

This step will be needed either if we are using raw video frames or video features.

    Script name:
        generate_descriptions_lists.py
    Description:
        Extracts and counts the available descriptions for each video.
    Output:
        - A file per split with the suffix _descriptions.txt. 
            Containing a list of descriptions for all videos.
        - A file per split with the suffix _descriptions_counts.npy. 
            Containing a python list with the counts of descriptions per video.
        The output will be stored in ./Annotations.


## Image lists generation

This step will be needed if we are using raw video frames only.

    Script name:
        generate_img_lists.py
    Description:
        Lists and counts the frames belonging to each video.
    Output:
        - A file per split with the suffix _imgs_list.txt. 
            Containing the list of frames for all videos.
        - A file per split with the suffix _imgs_counts.txt. 
            Containing a list of frame counts per video.
        The output will be stored in ./Annotations.


## Image features generation
    
This step will be needed if we are using image features only. The number of feature vectors per video does not need to match the number of frames. 

    Script name:
        generate_features_lists.py
    Description:
        Stores each feature vector contained in the corresponding .Features/[split_name]_[name_feat].csv in a separate .npy file and counts them.
    Output:
        - A file per split with the suffix _feat_list.txt.
            Containing the path to each feature vector.
        - A file per split with the suffix _feat_counts.txt.
            Containing the counts of vectors per video.
        The output .txt files will be stored in ./Annotations/[name_feat]/. And the .npy files in ./Features/[name_feat]/
            