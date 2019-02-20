import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def check_params(params):
    """
    Checks some typical parameters and warns if something wrong was specified.
    :param params: Model instance on which to apply the callback.
    :return: None
    """

    if params['TRG_PRETRAINED_VECTORS'] and params['TRG_PRETRAINED_VECTORS'][:-1] != '.npy':
        logger.warn('It seems that the pretrained word vectors provided for the target text are not in npy format.'
                    'You should preprocess the word embeddings with the "utils/preprocess_*_word_vectors.py script.')
    if not params['PAD_ON_BATCH']:
        logger.warn('It is HIGHLY recommended to set the option "PAD_ON_BATCH = True."')

    if 'from_logits' in params.get('LOSS', 'categorical_crossentropy'):
        if params.get('CLASSIFIER_ACTIVATION', 'softmax'):
            params['CLASSIFIER_ACTIVATION'] = None

    if params.get('LABEL_SMOOTHING', 0.) and 'sparse' in params.get('LOSS', 'categorical_crossentropy'):
        logger.warn('Label smoothing with sparse outputs is still unimplemented')

    return params
