# Automatic video captioning system

## Features: 

 * Attention model over the input sequence of frames
 * Peeked decoder LSTM: The previously generated word is an input of the current LSTM timestep
 * MLPs for initializing the LSTM hidden and memory state
 * Beam search decoding

## Instructions:

Assuming you have a dataset and features extracted from the video frames:
 
 1) Prepare data:
 
   ``
 python data_engine/subsample_frames_features.py
 ``
  ``
 python data_engine/generate_features_lists.py
 ``
  ``
 python data_engine/generate_descriptions_lists.py
 ``

2) Prepare the inputs/outputs of your model in `data_engine/prepare_data.py`
  
3) Set a model configuration in  `config.py` 
 
4) Train!:

  ``
 python main.py
 ``
