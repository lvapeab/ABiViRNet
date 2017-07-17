def load_parameters():
    """
        Loads the defined parameters
    """
    # Input data params
    DATA_ROOT_PATH = '/media/HDD_2TB/DATASETS/MSVD/'            # Root path to the data
    # preprocessed features
    DATASET_NAME = 'MSVD_features'                              # Dataset name

    # Input data
    INPUT_DATA_TYPE = 'video-features'                          # 'video-features' or 'video'
    NUM_FRAMES = 26                                             # fixed number of input frames per video
    
    #### Features from video frames
    FRAMES_LIST_FILES = {'train': 'Annotations/%s/train_feat_list.txt',                 # Feature frames list files
                         'val': 'Annotations/%s/val_feat_list.txt',
                         'test': 'Annotations/%s/test_feat_list.txt',
                        }
    FRAMES_COUNTS_FILES = {  'train': 'Annotations/%s/train_feat_counts.txt',           # Frames counts files
                             'val': 'Annotations/%s/val_feat_counts.txt',
                             'test': 'Annotations/%s/test_feat_counts.txt',
                          }
    FEATURE_NAMES = ['ImageNet'] # append '_L2' at the end of each feature type if using their L2 version

    # Output data
    DESCRIPTION_FILES = {'train': 'Annotations/train_descriptions.txt',                 # Description files
                         'val': 'Annotations/val_descriptions.txt',
                         'test': 'Annotations/test_descriptions.txt',
                        }
    DESCRIPTION_COUNTS_FILES = { 'train': 'Annotations/train_descriptions_counts.npy',  # Description counts files
                                 'val': 'Annotations/val_descriptions_counts.npy',
                                 'test': 'Annotations/test_descriptions_counts.npy',
                               }
    
    # Dataset parameters
    INPUTS_IDS_DATASET = ['video', 'state_below']   # Corresponding inputs of the dataset
    OUTPUTS_IDS_DATASET = ['description']           # Corresponding outputs of the dataset
    INPUTS_IDS_MODEL = ['video', 'state_below']     # Corresponding inputs of the built model
    OUTPUTS_IDS_MODEL = ['description']             # Corresponding outputs of the built model


    # Evaluation params
    METRICS = ['coco']  # Metric used for evaluating model after each epoch (leave empty if only prediction is required)
    EVAL_ON_SETS = ['val', 'test']                # Possible values: 'train', 'val' and 'test' (external evaluator)
    EVAL_ON_SETS_KERAS = []                       # Possible values: 'train', 'val' and 'test' (Keras' evaluator)
    START_EVAL_ON_EPOCH = 1                       # First epoch where the model will be evaluated
    EVAL_EACH_EPOCHS = True                       # Select whether evaluate between N epochs or N updates
    EVAL_EACH = 1                                 # Sets the evaluation frequency (epochs or updates)

    # Search parameters
    SAMPLING = 'max_likelihood'                   # Possible values: multinomial or max_likelihood (recommended)
    TEMPERATURE = 1                               # Multinomial sampling parameter
    BEAM_SEARCH = True                            # Switches on-off the beam search procedure
    BEAM_SIZE = 10                                 # Beam size (in case of BEAM_SEARCH == True)
    OPTIMIZED_SEARCH = True                       # Compute annotations only a single time per sample
    NORMALIZE_SAMPLING = True                     # Normalize hypotheses scores according to their length
    ALPHA_FACTOR = .6                             # Normalization according to length**ALPHA_FACTOR
                                                  # (see: arxiv.org/abs/1609.08144)

    # Sampling params: Show some samples during training
    SAMPLE_ON_SETS = ['train', 'val']             # Possible values: 'train', 'val' and 'test'
    N_SAMPLES = 5                                 # Number of samples generated
    START_SAMPLING_ON_EPOCH = 1                   # First epoch where the model will be evaluated
    SAMPLE_EACH_UPDATES = 450                     # Sampling frequency (default 450)

    # Word representation params
    TOKENIZATION_METHOD = 'tokenize_icann'        # Select which tokenization we'll apply:
                                                  #  tokenize_basic, tokenize_aggressive, tokenize_soft,
                                                  #  tokenize_icann or tokenize_questions

    FILL = 'end'                                  # whether we fill the 'end' or the 'start' of the sentence with 0s
    TRG_LAN = 'en'                                # Language of the outputs (mainly used for the Meteor evaluator)
    PAD_ON_BATCH = True                           # Whether we take as many timesteps as the longes sequence of the batch
                                                  # or a fixed size (MAX_OUTPUT_TEXT_LEN)

    # Input image parameters
    DATA_AUGMENTATION = False                      # Apply data augmentation on input data (noise on features)
    IMG_FEAT_SIZE = 1024                           # Size of the image features

    # Output text parameters
    OUTPUT_VOCABULARY_SIZE = 0                    # Size of the input vocabulary. Set to 0 for using all,
                                                  # otherwise it will be truncated to these most frequent words.
    MAX_OUTPUT_TEXT_LEN = 30                      # Maximum length of the output sequence
                                                  # set to 0 if we want to use the whole answer as a single class
    MAX_OUTPUT_TEXT_LEN_TEST = 120                # Maximum length of the output sequence during test time
    MIN_OCCURRENCES_VOCAB = 0                     # Minimum number of occurrences allowed for the words in the vocabulay.

    # Optimizer parameters (see model.compile() function)
    LOSS = 'categorical_crossentropy'
    CLASSIFIER_ACTIVATION = 'softmax'

    OPTIMIZER = 'Adam'                            # Optimizer
    LR = 0.001                                    # Learning rate. Recommended values - Adam 0.001 - Adadelta 1.0
    CLIP_C = 1.                                   # During training, clip L2 norm of gradients to this value (0. means deactivated)
    CLIP_V = 0.                                   # During training, clip absolute value of gradients to this value (0. means deactivated)
    SAMPLE_WEIGHTS = True                         # Select whether we use a weights matrix (mask) for the data outputs
    LR_DECAY = None                               # Minimum number of epochs before the next LR decay. Set to None if don't want to decay the learning rate
    LR_GAMMA = 0.8                                # Multiplier used for decreasing the LR

    # Training parameters
    MAX_EPOCH = 50                                # Stop when computed this number of epochs
    BATCH_SIZE = 64                               # ABiViRNet trained with BATCH_SIZE = 64

    HOMOGENEOUS_BATCHES = False                   # Use batches with homogeneous output lengths for every minibatch (Possibly buggy!)
    PARALLEL_LOADERS = 8                          # Parallel data batch loaders
    EPOCHS_FOR_SAVE = 1                           # Number of epochs between model saves
    WRITE_VALID_SAMPLES = True                    # Write valid samples in file
    SAVE_EACH_EVALUATION = True                   # Save each time we evaluate the model

    # Early stop parameters
    EARLY_STOP = True                             # Turns on/off the early stop protocol
    PATIENCE = 10                                 # We'll stop if the val cd  does not improve after this
                                                  # number of evaluations
    STOP_METRIC = 'Bleu_4'                        # Metric for the stop

    # Model parameters
    MODEL_TYPE = 'ABiVirNet'
    RNN_TYPE = 'LSTM'                             # RNN unit type ('LSTM' and 'GRU' supported)

    # Input text parameters
    TARGET_TEXT_EMBEDDING_SIZE = 420              # Source language word embedding size.
    TRG_PRETRAINED_VECTORS = None                 # Path to pretrained vectors. (e.g. DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % TRG_LAN)
                                                  # Set to None if you don't want to use pretrained vectors.
                                                  # When using pretrained word embeddings, the size of the pretrained word embeddings must match with the word embeddings size.
    TRG_PRETRAINED_VECTORS_TRAINABLE = True       # Finetune or not the target word embedding vectors.

    # Encoder configuration
    ENCODER_HIDDEN_SIZE = 600                     # For models with RNN encoder
    BIDIRECTIONAL_ENCODER = True                  # Use bidirectional encoder
    N_LAYERS_ENCODER = 1                          # Stack this number of encoding layers
    BIDIRECTIONAL_DEEP_ENCODER = True             # Use bidirectional encoder in all encoding layers

    DECODER_HIDDEN_SIZE = 484   # For models with LSTM decoder

    IMG_EMBEDDING_LAYERS = []  # FC layers for visual embedding
                               # Here we should specify the activation function and the output dimension
                               # (e.g IMG_EMBEDDING_LAYERS = [('linear', 1024)]

    # Fully-Connected layers for initializing the first RNN state
    #       Here we should only specify the activation function of each layer
    #       (as they have a potentially fixed size)
    #       (e.g INIT_LAYERS = ['tanh', 'relu'])
    INIT_LAYERS = ['tanh']

    # Additional Fully-Connected layers's sizes applied before softmax.
    #       Here we should specify the activation function and the output dimension
    #       (e.g DEEP_OUTPUT_LAYERS = [('tanh', 600), ('relu', 400), ('relu', 200)])
    DEEP_OUTPUT_LAYERS = [('linear', TARGET_TEXT_EMBEDDING_SIZE)]

    # Regularizers
    WEIGHT_DECAY = 1e-4                           # L2 regularization
    RECURRENT_WEIGHT_DECAY = 0.                   # L2 regularization in recurrent layers

    USE_DROPOUT = False                           # Use dropout
    DROPOUT_P = 0.5                               # Percentage of units to drop

    USE_RECURRENT_DROPOUT = False                 # Use dropout in recurrent layers # DANGEROUS!
    RECURRENT_DROPOUT_P = 0.5                     # Percentage of units to drop in recurrent layers

    USE_NOISE = True                              # Use gaussian noise during training
    NOISE_AMOUNT = 0.01                           # Amount of noise

    USE_BATCH_NORMALIZATION = True                # If True it is recommended to deactivate Dropout
    BATCH_NORMALIZATION_MODE = 1                  # See documentation in Keras' BN

    USE_PRELU = False                             # use PReLU activations as regularizer
    USE_L2 = False                                # L2 normalization on the features

    # Results plot and models storing parameters
    EXTRA_NAME = ''                    # This will be appended to the end of the model name
    MODEL_NAME = DATASET_NAME + '_' + MODEL_TYPE +\
                 '_txtemb_' + str(TARGET_TEXT_EMBEDDING_SIZE) + \
                 '_imgemb_' + '_'.join([layer[0] for layer in IMG_EMBEDDING_LAYERS]) + \
                 '_lstmenc_' + str(ENCODER_HIDDEN_SIZE) + \
                 '_lstm_' + str(DECODER_HIDDEN_SIZE) + \
                 '_deepout_' + '_'.join([layer[0] for layer in DEEP_OUTPUT_LAYERS]) + \
                 '_' + OPTIMIZER

    MODEL_NAME += EXTRA_NAME

    STORE_PATH = 'trained_models/' + MODEL_NAME + '/'  # Models and evaluation results will be stored here
    DATASET_STORE_PATH = 'datasets/'                   # Dataset instance will be stored here

    SAMPLING_SAVE_MODE = 'list'                        # 'list' or 'vqa'
    VERBOSE = 1                                        # Verbosity level
    RELOAD = 0                                         # If 0 start training from scratch, otherwise the model
                                                       # Saved on epoch 'RELOAD' will be used
    REBUILD_DATASET = True                             # Build again or use stored instance
    MODE = 'training'                                  # 'training' or 'sampling' (if 'sampling' then RELOAD must
                                                       # be greater than 0 and EVAL_ON_SETS will be used)

    # Extra parameters for special trainings
    TRAIN_ON_TRAINVAL = False                          # train the model on both training and validation sets combined
    FORCE_RELOAD_VOCABULARY = False                    # force building a new vocabulary from the training samples
                                                       # applicable if RELOAD > 1

    # ================================================ #
    parameters = locals().copy()
    return parameters
