__version__ = "2.3.0"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging

    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    absl.logging._warn_preinit_stderr = False
except:
    pass

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Files and general utilities
from .file_utils import (TRANSFORMERS_CACHE, PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE,
                         cached_path, add_start_docstrings, add_end_docstrings,
                         WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, CONFIG_NAME, MODEL_CARD_NAME,
                         is_tf_available, is_torch_available)

from .data import (InputExample, InputFeatures, DataProcessor,
                   SingleSentenceClassificationProcessor,
                   glue_output_modes, glue_convert_examples_to_features,
                   glue_processors, glue_tasks_num_labels,
                   squad_convert_examples_to_features, SquadFeatures, SquadExample, SquadV1Processor,
                   SquadV2Processor, glue_compute_metrics, xnli_compute_metrics)

# coqa_convert_examples_to_features, CoQAFeatures, CoqaExample, read_coqa_examples)

# Modeling
if is_torch_available():

    from .modeling_albert import (AlbertPreTrainedModel, AlbertModel, AlbertForMaskedLM,
                                  AlbertForSequenceClassification,
                                  AlbertForQuestionAnswering, AlbertForQuestionAnsweringAV,
                                  AlbertForQuestionAnsweringCLS, AlbertForQuestionAnsweringAVPool,
                                  AlbertForQuestionAnsweringAVReg,
                                  AlbertForQuestionAnsweringAVPoolBCE,
                                  AlbertForQuestionAnsweringAVPoolBCEv3,
                                  AlbertForQuestionAnsweringDep, AlbertForQuestionAnsweringDep2,
                                  AlbertForQuestionAnsweringSeqTrm, AlbertForQuestionAnsweringSeqSC,
                                  AlbertForQuestionAnsweringAVPoolXL,
                                  load_tf_weights_in_albert, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP)

    # Optimization
    from .optimization import (AdamW, get_constant_schedule, get_constant_schedule_with_warmup,
                               get_cosine_schedule_with_warmup,
                               get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup)

# Pipelines
from .pipelines import pipeline, PipelineDataFormat, CsvPipelineDataFormat, JsonPipelineDataFormat, \
    PipedPipelineDataFormat, \
    Pipeline, FeatureExtractionPipeline, QuestionAnsweringPipeline, NerPipeline, TextClassificationPipeline

if not is_tf_available() and not is_torch_available():
    logger.warning("Neither PyTorch nor TensorFlow >= 2.0 have been found."
                   "Models won't be available and only tokenizers, configuration"
                   "and file/data utilities can be used.")
