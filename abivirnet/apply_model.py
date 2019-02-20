# -*- coding: utf-8 -*-
from __future__ import print_function
try:
    import itertools.imap as map
except ImportError:
    pass
import logging
from keras_wrapper.extra.read_write import list2file, nbest2file, list2stdout, numpy2file, pkl2dict

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)



def apply_Video_model(params):
    """
        Function for using a previously trained model for sampling.
    """

    ########### Load data
    dataset = build_dataset(params)
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    ###########


    ########### Load model
    video_model = loadModel(params['STORE_PATH'], params['RELOAD'])
    video_model.setOptimizer()
    ###########


    ########### Apply sampling
    extra_vars = dict()
    extra_vars['tokenize_f'] = eval('dataset.' + params['TOKENIZATION_METHOD'])
    extra_vars['language'] = params.get('TRG_LAN', 'en')

    for s in params["EVAL_ON_SETS"]:

        # Apply model predictions
        params_prediction = {'max_batch_size': params['BATCH_SIZE'],
                             'n_parallel_loaders': params['PARALLEL_LOADERS'],
                             'predict_on_sets': [s]}

        # Convert predictions into sentences
        vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']

        if params['BEAM_SEARCH']:
            params_prediction['beam_size'] = params['BEAM_SIZE']
            params_prediction['maxlen'] = params['MAX_OUTPUT_TEXT_LEN_TEST']
            params_prediction['optimized_search'] = params['OPTIMIZED_SEARCH']
            params_prediction['model_inputs'] = params['INPUTS_IDS_MODEL']
            params_prediction['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            params_prediction['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            params_prediction['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            params_prediction['normalize_probs'] = params['NORMALIZE_SAMPLING']

            params_prediction['alpha_factor'] = params['ALPHA_FACTOR']
            predictions = video_model.predictBeamSearchNet(dataset, params_prediction)[s]
            predictions = video_model.decode_predictions_beam_search(predictions,
                                                                     vocab,
                                                                     verbose=params['VERBOSE'])
        else:
            predictions = video_model.predictNet(dataset, params_prediction)[s]
            predictions = video_model.decode_predictions(predictions, 1,  # always set temperature to 1
                                                         vocab, params['SAMPLING'], verbose=params['VERBOSE'])

        # Store result
        filepath = video_model.model_path + '/' + s + '_sampling.pred'  # results file
        if params['SAMPLING_SAVE_MODE'] == 'list':
            list2file(filepath, predictions)
        else:
            raise Exception, 'Only "list" is allowed in "SAMPLING_SAVE_MODE"'

        # Evaluate if any metric in params['METRICS']
        for metric in params['METRICS']:
            logging.info('Evaluating on metric ' + metric)
            filepath = video_model.model_path + '/' + s + '_sampling.' + metric  # results file

            # Evaluate on the chosen metric
            extra_vars[s] = dict()
            extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
            metrics = evaluation.select[metric](
                pred_list=predictions,
                verbose=1,
                extra_vars=extra_vars,
                split=s)

            # Print results to file
            with open(filepath, 'w') as f:
                header = ''
                line = ''
                for metric_ in sorted(metrics):
                    value = metrics[metric_]
                    header += metric_ + ','
                    line += str(value) + ','
                f.write(header + '\n')
                f.write(line + '\n')
            logging.info('Done evaluating on metric ' + metric)
