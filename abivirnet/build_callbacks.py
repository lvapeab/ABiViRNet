# -*- coding: utf-8 -*-
from keras_wrapper.extra.callbacks import *

def buildCallbacks(params, model, dataset):
    """
    Builds the selected set of callbacks run during the training of the model.

    :param params: Dictionary of network hyperparameters.
    :param model: Model instance on which to apply the callback.
    :param dataset: Dataset instance on which to apply the callback.
    :return:
    """

    callbacks = []

    if params['METRICS']:
        # Evaluate training
        extra_vars = {'language': params.get('TRG_LAN', 'en'),
                      'n_parallel_loaders': params['PARALLEL_LOADERS'],
                      'tokenize_f': eval('dataset.' + params['TOKENIZATION_METHOD'])}
        vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
        for s in params['EVAL_ON_SETS']:
            extra_vars[s] = dict()
            extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
        if params['BEAM_SIZE']:
            extra_vars['beam_size'] = params.get('BEAM_SIZE', 6)
            extra_vars['state_below_index'] = params.get('BEAM_SEARCH_COND_INPUT', -1)
            extra_vars['maxlen'] = params.get('MAX_OUTPUT_TEXT_LEN_TEST', 30)
            extra_vars['optimized_search'] = params.get('OPTIMIZED_SEARCH', True)
            extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
            extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            extra_vars['normalize_probs'] = params.get('NORMALIZE_SAMPLING', False)
            extra_vars['alpha_factor'] = params.get('ALPHA_FACTOR', 1.)

        callback_metric = PrintPerformanceMetricOnEpochEndOrEachNUpdates(model,
                                                                         dataset,
                                                                         gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                                                         metric_name=params['METRICS'],
                                                                         set_name=params['EVAL_ON_SETS'],
                                                                         batch_size=params['BATCH_SIZE'],
                                                                         each_n_epochs=params['EVAL_EACH'],
                                                                         extra_vars=extra_vars,
                                                                         reload_epoch=params['RELOAD'],
                                                                         is_text=True,
                                                                         input_text_id=None,
                                                                         index2word_y=vocab,
                                                                         index2word_x=None,
                                                                         sampling_type=params['SAMPLING'],
                                                                         beam_search=params['BEAM_SEARCH'],
                                                                         save_path=model.model_path,
                                                                         start_eval_on_epoch=params[
                                                                             'START_EVAL_ON_EPOCH'],
                                                                         write_samples=True,
                                                                         write_type=params['SAMPLING_SAVE_MODE'],
                                                                         eval_on_epochs=params['EVAL_EACH_EPOCHS'],
                                                                         save_each_evaluation=params[
                                                                             'SAVE_EACH_EVALUATION'],
                                                                         verbose=params['VERBOSE'])
        callbacks.append(callback_metric)

        if params['SAMPLE_ON_SETS']:
            # Evaluate sampling
            extra_vars = {'language': params['TRG_LAN'], 'n_parallel_loaders': params['PARALLEL_LOADERS']}
            vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
            for s in params['EVAL_ON_SETS']:
                extra_vars[s] = dict()
                extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
                extra_vars[s]['tokenize_f'] = eval('dataset.' + params['TOKENIZATION_METHOD'])
            if params['BEAM_SIZE']:
                extra_vars['beam_size'] = params['BEAM_SIZE']
                extra_vars['state_below_index'] = params.get('BEAM_SEARCH_COND_INPUT', -1)
                extra_vars['maxlen'] = params['MAX_OUTPUT_TEXT_LEN_TEST']
                extra_vars['optimized_search'] = params['OPTIMIZED_SEARCH']
                extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
                extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
                extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
                extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
                extra_vars['normalize_probs'] = params['NORMALIZE_SAMPLING']
                extra_vars['alpha_factor'] = params['ALPHA_FACTOR']

            callback_sampling = SampleEachNUpdates(model,
                                                   dataset,
                                                   gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                                   set_name=params['SAMPLE_ON_SETS'],
                                                   n_samples=params['N_SAMPLES'],
                                                   each_n_updates=params['SAMPLE_EACH_UPDATES'],
                                                   extra_vars=extra_vars,
                                                   reload_epoch=params['RELOAD'],
                                                   is_text=True,
                                                   print_sources=False,
                                                   index2word_x=None,  # text info
                                                   index2word_y=vocab,  # text info
                                                   sampling_type=params['SAMPLING'],  # text info
                                                   beam_search=params['BEAM_SEARCH'],
                                                   start_sampling_on_epoch=params['START_SAMPLING_ON_EPOCH'],
                                                   verbose=params['VERBOSE'])
            callbacks.append(callback_sampling)

    return callbacks