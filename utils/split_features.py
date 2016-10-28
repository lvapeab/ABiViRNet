import numpy as np



def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=np.float32):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data



base_path =  '/media/HDD_2TB/DATASETS/MSVD/Features/'
feature = 'ImageNetFV_Places_C3Dfc8'
out_feature = 'ImageNetFV'




for split in ['train', 'val', 'test']:
    print "Loading %s features" %str(split + '_' + feature)
    #feats = np.genfromtxt(open(base_path + split + '_' + feature + "_features.csv", "rb"), delimiter=",", dtype='float32')
    feats = iter_loadtxt(base_path + split + '_' + feature + "_features.csv")
    new_feats = feats[:, :1024] # Modify this instruction to get the desired features!
    print "Saving %s features" %str(split + '_' + feature)
    np.savetxt(base_path + split + '_' + out_feature + "_features.csv", new_feats, delimiter=",")