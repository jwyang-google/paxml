"""Language Model configurations on the T5/C4 dataset."""

import functools
import math
from typing import Dict, List, Optional, Sequence

from absl import logging
import fiddle as fdl
import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import experiment_registry
from paxml import seqio_input
from paxml import tasks_lib
from paxml import trainer_lib
from paxml.tasks.lm import model_params
from paxml.tasks.lm.params import lm_cloud
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis.layers import transformers
import seqio
import t5.data
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics

WeightInit = base_layer.WeightInit

GPT_SPM_PATH = (
    'gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model'
)
GPT_EOS_ID = 1
GPT_VOCABULARY = t5.data.SentencePieceVocabulary(GPT_SPM_PATH)
PASS_THROUGH_VOCABULARY = t5.data.PassThroughVocabulary(size=50257)

C4_GPT_TRAIN_FEATURES_LM = {
    'inputs': t5.data.Feature(vocabulary=GPT_VOCABULARY, add_eos=False),
    'targets': t5.data.Feature(vocabulary=GPT_VOCABULARY, add_eos=False)
}

C4_GPT_EVAL_FEATURES_LM = {
    'inputs': t5.data.Feature(
        vocabulary=PASS_THROUGH_VOCABULARY, add_eos=False
    ),
    'targets': t5.data.Feature(
        vocabulary=PASS_THROUGH_VOCABULARY, add_eos=False
    )
}
TRAIN_DATADIR = 'gs://test-example-123/cnndm/'
EVAL_DATADIR = 'gs://test-example-123/cnndm/'
CNN_DATADIR = 'gs://test-example-123/datasets'

class TaskRegistry(t5.data.TaskRegistry):
  """Task registry with extra tracking."""

  TASK_NAMES = []

  @classmethod
  def add_versioned_tfds_task(cls,
                              name: str,
                              *,
                              versions: List[str],
                              pinned_version: Optional[str] = None,
                              tfds_name: str,
                              tfds_data_dir: Optional[str] = None,
                              **kwargs) -> List[seqio.Task]:
    tasks = []
    for version in versions:
      tasks.append(
          cls.add(
              f'{name}_{version}',
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    if pinned_version is not None:
      tasks.append(
          cls.add(
              name,
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{pinned_version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    return tasks

# C4 corpus for language model pretraining
TaskRegistry.add_versioned_tfds_task(
    name="cnn_dailymail_v001",
    versions=['3.4.0'],
    pinned_version='3.4.0',
    tfds_name='cnn_dailymail',
    tfds_data_dir=CNN_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.summarize,
            article_key='article',
            summary_key='highlights',
        ),
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=C4_GPT_TRAIN_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=10000,
)

TaskRegistry.add_versioned_tfds_task(
    name="cnn_dailymail_v001_eval",
    versions=['3.4.0'],
    pinned_version='3.4.0',
    tfds_name='cnn_dailymail',
    tfds_data_dir=CNN_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.summarize,
            article_key='article',
            summary_key='highlights',
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=C4_GPT_TRAIN_FEATURES_LM,
    metric_fns=[t5_metrics.bleu, t5_metrics.rouge],
    shuffle_buffer_size=10000,
)

class CNNDMUnsupervisedDataset(base_experiment.BaseExperiment):
  """Used for training Baseline ULM."""

  PERCORE_BATCH_SIZE = 1
  PERCORE_EVAL_BATCH_SIZE = None
  MAX_SEQ_LEN = 2048
  TRAINING_SEED = 9876
  TRAINING_NUM_BATCHES_TO_SKIP = None
  DECODE_INPUTS_LENGTH = MAX_SEQ_LEN
  OUTPUTS_LENGTH = 128
  def _dataset_common(self, is_training) -> pax_fiddle.Config[base_input.BaseInput]:
    if is_training:
      percore_batch_size = self.PERCORE_BATCH_SIZE
    else:
      if self.PERCORE_EVAL_BATCH_SIZE is not None:
        percore_batch_size = self.PERCORE_EVAL_BATCH_SIZE
      else:
        percore_batch_size = self.PERCORE_BATCH_SIZE

    num_local_devices = jax.local_device_count()
    global_batch_size = int(
        percore_batch_size * num_local_devices * jax.process_count() + 1e-6
    )
    if percore_batch_size >= 1:
      assert global_batch_size % num_local_devices == 0
      batch_size_per_process = int(
          math.ceil(percore_batch_size) * num_local_devices + 1e-6
      )
      num_infeed_hosts = global_batch_size // batch_size_per_process
    else:
      if jax.process_count() > 1:
        assert global_batch_size % num_local_devices == 0
        batch_size_per_process = num_local_devices
        num_infeed_hosts = global_batch_size // batch_size_per_process
      else:
        batch_size_per_process = int(
            percore_batch_size * num_local_devices + 1e-6
        )
        num_infeed_hosts = 1
    seed = None
    if is_training:
      seed = self.TRAINING_SEED
      # TODO(sgpyc): enable sync of seeds across hosts, currently the
      # following failed because of "sync_global_devices name mismatch"
      # seed = jnp.int32(multihost_utils.broadcast_one_to_all(seed))
      logging.info('Train input seed: %s',
                   'None' if seed is None else seed)
    p = pax_fiddle.Config(
        seqio_input.SeqIOInput,
        name='cnnTrain' if is_training else 'cnnValidation',
        mixture_name='cnn_dailymail_v001_3.4.0' if is_training else 'cnn_dailymail_v001_eval_3.4.0',
        split_name='train' if is_training else 'validation',
        task_feature_lengths={
          'inputs': self.MAX_SEQ_LEN,
          'targets': 256},
        use_cached=False,
        repeat=True if is_training else False,
        feature_converter=seqio_input.LanguageModelFeatures(
            pack=False
        ),
        is_training=is_training,
        input_random_seed=(seed if is_training else 4321),
        batch_size=batch_size_per_process,
        drop_remainder=True if is_training else False,
        num_batches_to_skip=self.TRAINING_NUM_BATCHES_TO_SKIP,
        num_infeed_hosts=num_infeed_hosts,
        reset_for_eval=False if is_training else True,
        annotate_padding_fields=True,
    )
    return p

  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns a list of dataset parameters."""
    return [
        self._dataset_common(is_training=True),
        self._dataset_common(is_training=False),
    ]

  def _add_decoder_dataset(
      self, dataset_mixture_name, split_name='validation'
  ) -> Sequence[pax_fiddle.Config[base_input.BaseInput]]:
    if self.PERCORE_EVAL_BATCH_SIZE is not None:
      percore_batch_size = self.PERCORE_EVAL_BATCH_SIZE
    else:
      percore_batch_size = self.PERCORE_BATCH_SIZE
        
    batch_size = percore_batch_size * jax.local_device_count()
    feature_lengths = {
        'inputs': self.DECODE_INPUTS_LENGTH,
        'targets': self.OUTPUTS_LENGTH
    }
    return seqio_input.get_eval_hparams_for_seqio(
        dataset_mixture_name,
        batch_size,
        feature_lengths,
        seed=75303,
        metric_type=seqio_input.MetricType.PREDICT,
        split_name=split_name,
        feature_converter= seqio_input.LanguageModelFeatures(
            pack=False
        ))

  def decoder_datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    datasets = []
    datasets += self._add_decoder_dataset('cnn_dailymail_v001_eval_3.4.0')
    return datasets
    

  
def set_adam_and_learning_rate_schedule(
    cls,
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
) -> pax_fiddle.Config[tasks_lib.SingleTask]:
  """Sets the Adam optimizer and the learning rate schedule."""
  lp = task_p.train.learner
  lp.loss_name = 'total_loss'
  lp.optimizer = pax_fiddle.Config(
      optimizers.Adam,
      beta1=cls.ADAM_BETA1 if cls.ADAM_BETA1 else 0.9,
      beta2=cls.ADAM_BETA2 if cls.ADAM_BETA2 else 0.999,
      weight_decay=cls.WEIGHT_DECAY if cls.WEIGHT_DECAY else 0.0,
      epsilon=cls.ADAM_EPSILON if cls.ADAM_EPSILON else 1e-6,
      epsilon_root=cls.ADAM_EPSILON_ROOT if cls.ADAM_EPSILON_ROOT else 0.0,
      clip_gradient_norm_to_value=cls.CLIP_GRADIENT_NORM_TO_VALUE
      if cls.CLIP_GRADIENT_NORM_TO_VALUE
      else 5.0,
      clip_threshold=cls.ADAM_CLIP_THRESHOLD
      if cls.ADAM_CLIP_THRESHOLD
      else 1.0,
  )

  if hasattr(cls, 'PERCORE_BATCH_SIZE'):
    print(cls.PERCORE_BATCH_SIZE)
    global_batch_size = int(cls.PERCORE_BATCH_SIZE * jax.device_count() + 1e-6)
    assert global_batch_size > 0
    assert global_batch_size <= 8192
  else:
    global_batch_size = None

  if cls.LEARNING_RATE is not None:
    lp.optimizer.learning_rate = cls.LEARNING_RATE
  else:
    assert global_batch_size is not None
    if global_batch_size <= 3584:
      lp.optimizer.learning_rate = 2e-5
    else:
      lp.optimizer.learning_rate = 3e-5

  if cls.LR_SCHEDULE == 'linear_rampup_exponential_decay':
    lp.optimizer.lr_schedule = pax_fiddle.Config(
        schedules.LinearRampupExponentialDecay,
        warmup_steps=cls.LR_LRED_WARMUP,
        decay_start=cls.LR_LRED_DECAY_START,
        decay_end=cls.LR_LRED_DECAY_END,
        min_ratio=cls.LR_LRED_MIN_RATIO,
        max=cls.LR_LRED_MAX,
    )
  elif cls.LR_SCHEDULE == 'linear_rampup_cosine_decay':
    if cls.LR_COS_WARMUP is not None:
      warmup_steps = cls.LR_COS_WARMUP
    else:
      assert global_batch_size is not None
      warmup_steps = math.ceil(265.0 * 1536 / global_batch_size - 1e-6)
      assert warmup_steps > 0

    if cls.LR_COS_DECAY_START is not None:
      decay_start_step = cls.LR_COS_DECAY_START
    else:
      decay_start_step = warmup_steps + 1

    if cls.LR_COS_DECAY_END is not None:
      decay_end_step = cls.LR_COS_DECAY_END
    else:
      assert global_batch_size is not None
      decay_end_step = math.ceil(108600.0 * 1536 / global_batch_size - 1e-6)
      assert decay_end_step > 0

    lp.optimizer.lr_schedule = pax_fiddle.Config(
        schedules.LinearRampupCosineDecay,
        warmup_steps=warmup_steps,
        decay_start=decay_start_step,
        decay_end=decay_end_step,
        min_ratio=cls.LR_COS_MIN_RATIO,
        max=cls.LR_COS_MAX,
    )
  else:
    raise NotImplementedError(
        f'Learning rate schedule {cls.LR_SCHEDULE} is not supported.'
    )

  return task_p


class TransformerLmSpmdAdam(model_params.TransformerLmSpmdAdafactor):
  """Base SPMD Transformer LM configuration using Adam.

  Only things different from TransformerLmSpmdAdafactor are listed.
  """
  # architecture related
  NUM_LAYERS = 32
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.float32
  PACKED_INPUT = True
  USE_BIAS = False
  EMBEDDING_LOOKUP_STYLE = 'matmul'

  # optimizer related
  LEARNING_RATE = 1e-3
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.99
  ADAM_CLIP_THRESHOLD = 1.0
  ADAM_EPSILON = 1e-6
  ADAM_EPSILON_ROOT = 0.0

  # Learning rate schedule
  LR_SCHEDULE = 'linear_rampup_exponential_decay'
  LR_LRED_WARMUP = 4000
  LR_LRED_DECAY_START = 4001
  LR_LRED_DECAY_END = 300000
  LR_LRED_MIN_RATIO = 0.1
  LR_LRED_MAX = 1.0

  LR_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0
  LR_COS_WARMUP = 4000
  LR_COS_DECAY_START = 4001
  LR_COS_DECAY_END = 300000
  
  USE_REPEATED_LAYER = True
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT  # pytype: disable=attribute-error  # enable-nested-classes

    stacked_p = model_p.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
    if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if self.USE_REPEATED_LAYER:
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p


class TransformerLmSpmdPipelineAdam(
    model_params.TransformerLmSpmdPipelineAdafactor
):
  """Base pipelined SPMD Transformer LM configuration using Adam.

  Only things different from TransformerLmSpmdPipelineAdafactor are listed.
  """
  # architecture related
  NUM_LAYERS = 32
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.float32
  PACKED_INPUT = True
  USE_BIAS = False
  EMBEDDING_LOOKUP_STYLE = 'matmul'

  # optimizer related
  LEARNING_RATE = 1e-3
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.99
  ADAM_CLIP_THRESHOLD = 1.0
  ADAM_EPSILON = 1e-6
  ADAM_EPSILON_ROOT = 0.0

  # Learning rate schedule
  LR_SCHEDULE = 'linear_rampup_exponential_decay'
  LR_LRED_WARMUP = 4000
  LR_LRED_DECAY_START = 4001
  LR_LRED_DECAY_END = 300000
  LR_LRED_MIN_RATIO = 0.1
  LR_LRED_MAX = 1.0

  LR_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0
  LR_COS_WARMUP = 4000
  LR_COS_DECAY_START = 4001
  LR_COS_DECAY_END = 300000

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT  # pytype: disable=attribute-error  # enable-nested-classes

    stacked_p = model_p.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
    if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if self.USE_REPEATED_LAYER:
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p


@experiment_registry.register
class LmCloudSpmdAdam(TransformerLmSpmdAdam, lm_cloud.SyntheticDataset):
  """Base config for an SPMD model."""
  NUM_LAYERS = 2
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 2]


@experiment_registry.register
class LmCloudSpmdAdamLimitSteps(LmCloudSpmdAdam):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 4000
    return task_p


class EarlyStoppingFn(base_hyperparams.FiddleBaseParameterizable):
  r"""Early stopping function to log eval log_pplx and stop when reaching target.

  Attributes:
    target_log_pplx: target log pplx value to stop training when eval log pplx
      reaches this value.
  """

  target_log_pplx: Optional[float] = None

  def __call__(
      self,
      metrics: Dict[str, float],
      running_mode: trainer_lib.RunningMode,
      global_step: int,
      is_last_ckpt: bool,
  ) -> bool:
    """Returns True if run should be stopped early."""
    if 'eval_test_C4Validation/metrics/log_pplx' not in metrics.keys():
      return False
    log_pplx = metrics['eval_test_C4Validation/metrics/log_pplx']

    if log_pplx <= self.target_log_pplx:
      return True
    return False
  
@experiment_registry.register
class CNNDMSpmdAdam(TransformerLmSpmdAdam, 
                    CNNDMUnsupervisedDataset):
  r"""Base config for a decoder only transformer with pipeline."""
  NUM_LAYERS = 24
  NUM_HEADS = 32
  MODEL_DIMS = 2048
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 32128
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B
  CHECKPOINT_EVERY_N_STEPS = 1000

  PERCORE_BATCH_SIZE = 0.5  # 512 global batch size
  
  # Sub-class has to specify a mesh.
  MICROBATCH_SIZE = 2
  ICI_MESH_SHAPE = [2, 1, 2, 2]
  NUM_STAGES = 2
  EMB_W_DATA_DIMS = ('replica', 'data')

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.eos_id = (
        GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
    )
    model_p.decoder_tpl.seqlen = self.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p
  

 
@experiment_registry.register
class CNNDMSpmdRepeatedGpt3SmallAdam1X2X8(CNNDMSpmdAdam):
  """Small GPT-3 config in bf16 for 8 replicas with 512 global batch size.

  This was called GPT-3 XL in the GPT-3 paper, with 1.3B parameters.
  """
  NUM_STAGES = 1
  NUM_LAYERS = 16
  NUM_HEADS = 16
  MODEL_DIMS = 4096
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  PERCORE_BATCH_SIZE = 0.5  # 512 global batch size
  ICI_MESH_SHAPE = [1, 4, 4]
  DCN_MESH_SHAPE = [1, 1, 1]
  FPROP_DTYPE = jnp.bfloat16
  
  NUM_DECODE_BATCHES = 16
  DECODE_INTERVAL_STEPS = 50
  EVAL_INTERVAL_STEPS = 50
  PERCORE_EVAL_BATCH_SIZE = 4
  USE_REPEATED_LAYER = True
  
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 100
    task_p.train.decode_interval_steps = 50

    return task_p


@experiment_registry.register
class C4SpmdPipelineBLOOM6B(CNNDMSpmdRepeatedGpt3SmallAdam1X2X8):
  """BLOOM config for MLPerf inference reference."""
  PERCORE_BATCH_SIZE = 0.25

  VOCAB_SIZE = 250880
  NUM_LAYERS = 30
  NUM_HEADS = 32

  NUM_STAGES = 1
  ICI_MESH_SHAPE = [1, 4, 4] # [1, 1, 8, 2]
  MODEL_DIMS = 4096
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = 128

  # BLOOM model architecture
  TRAINABLE_POSITION_EMB = False
  USE_BIAS = True
  EMBEDDING_LOOKUP_STYLE = 'matmul'
  ATTEN_LOGIT_CAP = -1.0  # Disable logits cap in atten

  # bf16 data type
  FPROP_DTYPE = jnp.bfloat16
  
  # layernorm episilon
  LAYERNORM_EPSILON = 1e-6

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    configure_bloom_task(self, task_p)
    task_p.train.num_train_steps = 100
    task_p.train.decode_interval_steps = 50

    return task_p
  
@experiment_registry.register
class C4SpmdPipelineBLOOM6BFromJAX(C4SpmdPipelineBLOOM6B):
  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.init_from_checkpoint_rules = {
    'gs://jwyang-cloud-tpu-tutorial/BLOOM6B_ckpt_from_jax_pax_repated/':
        tasks_lib.CheckpointLoadingRules(
            task_p=C4SpmdPipelineBLOOM6B().task(),
            # Load all of TrainState.
            load_rules=[(r'(.*)', '{}')],
            load_step=False,
            load_opt_states=False,
            step=0,
            input_specs_provider_p=C4SpmdPipelineBLOOM6B().get_input_specs_provider_params()
        )
    }

    return task_p
  

def configure_bloom_task(
    cls,
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
) -> pax_fiddle.Config[tasks_lib.SingleTask]:
  """Returns task with gpt3 related configs."""
  model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes

  model_p.decoder_tpl.eos_id = (
      GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
  )
  model_p.decoder_tpl.seqlen = cls.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

  model_p.params_init = WeightInit.Gaussian(0.006)

  softmax_init = WeightInit.Gaussian(0.006)
  model_p.lm_tpl.softmax_tpl.params_init = softmax_init
  model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False
  model_p.lm_tpl.softmax_tpl.soft_cap_logits = None

  if cls.SEPARATE_EMBEDDING:
    model_p.lm_tpl.separate_embedding_tpl.scale_sqrt_depth = False
    model_p.lm_tpl.separate_embedding_tpl.lookup_style = (
        cls.EMBEDDING_LOOKUP_STYLE
    )
  else:
    model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = False
    model_p.lm_tpl.softmax_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE
  if cls.TRAINABLE_POSITION_EMB:
    model_p.lm_tpl.position_emb_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE

  stacked_p = model_p.lm_tpl.stacked_transformer_tpl
  if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
    stacked_p = stacked_p.pipeline_stage
  if issubclass(
      fdl.get_callable(stacked_p), transformers.StackedTransformerRepeated
  ):
    stacked_p = stacked_p.block
  transformer_layer_p = stacked_p.transformer_layer_params_tpl

  transformer_layer_p.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
  transformer_layer_p.tr_fflayer_tpl.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
  model_p.lm_tpl.final_ln_tpl.epsilon = cls.LAYERNORM_EPSILON
  transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
  transformer_layer_p.tr_atten_tpl.use_bias = True
  transformer_layer_p.tr_atten_tpl.atten_logit_cap = cls.ATTEN_LOGIT_CAP

  transformer_layer_p.tr_fflayer_tpl.activation_tpl.approximate = True

  for atten_p in (
      transformer_layer_p.tr_atten_tpl,
      transformer_layer_p.cross_atten_tpl,
  ):
    if atten_p is None:
      continue
    atten_wp = atten_p.weight_split_dims_mapping
    atten_wp.proj = ['data', 'mdl', None]

  # if task_p.early_stopping_fn is None:
  #   task_p.early_stopping_fn = pax_fiddle.Config(EarlyStoppingFn)
  #   task_p.early_stopping_fn.target_log_pplx = cls.TARGET_LOG_PPLX

  return task_p 







@experiment_registry.register
class CNNDMSpmdPipelineAdam(TransformerLmSpmdPipelineAdam, 
                            CNNDMUnsupervisedDataset):
  r"""Base config for a decoder only transformer with pipeline."""
  NUM_LAYERS = 24
  NUM_HEADS = 32
  MODEL_DIMS = 2048
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 32128
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B
  CHECKPOINT_EVERY_N_STEPS = 1000

  PERCORE_BATCH_SIZE = 0.5  # 512 global batch size
  
  # Sub-class has to specify a mesh.
  MICROBATCH_SIZE = 2
  ICI_MESH_SHAPE = [2, 1, 2, 2]
  NUM_STAGES = 2
  EMB_W_DATA_DIMS = ('replica', 'data')

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.eos_id = (
        GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
    )
    model_p.decoder_tpl.seqlen = self.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p
  
@experiment_registry.register
class CNNSpmdPipelineGpt3AdamMLPerfHPBS1p5k768Replicas(
    CNNDMSpmdPipelineAdam
):
  r"""GPT-3 config in fp32 for 768 replicas with 1536 global batch size.

  Following MLPerf training benchmarking HP requirements.
  """
  VOCAB_SIZE = 32000
  PERCORE_BATCH_SIZE = 2
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 12]
  # NUM_MICROBATCHS = 192
  MICROBATCH_SIZE = 8
  EVAL_INTERVAL_STEPS = 16
  CHECKPOINT_EVERY_N_STEPS = EVAL_INTERVAL_STEPS * 2
  STREAM_IO = False
  SEPARATE_EMBEDDING = False
  
@experiment_registry.register
class CNNDMSpmdPipelineGpt3SmallAdam8Replicas(CNNDMSpmdPipelineAdam):
  """Small GPT-3 config in bf16 for 8 replicas with 512 global batch size.

  This was called GPT-3 XL in the GPT-3 paper, with 1.3B parameters.
  """

  NUM_STAGES = 2
  NUM_LAYERS = 24
  NUM_HEADS = 24
  MODEL_DIMS = 3072
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = 128
  VOCAB_SIZE = 51200

  PERCORE_BATCH_SIZE = 64
  MICROBATCH_SIZE = 8
  FPROP_DTYPE = jnp.bfloat16
  LEARNING_RATE = 2.0e-4
  ICI_MESH_SHAPE = [2, 1, 2, 2]

  CHECKPOINT_MAX_TO_KEEP = 1000
  EVAL_INTERVAL_STEPS = 10
  SUMMARY_INTERVAL_STEPS = 5
  CHECKPOINT_EVERY_N_STEPS = 200
  
@experiment_registry.register
class CNNDMSpmdPipelineGpt3SmallAdam1X2X4(CNNDMSpmdPipelineAdam):
  """Small GPT-3 config in bf16 for 8 replicas with 512 global batch size.

  This was called GPT-3 XL in the GPT-3 paper, with 1.3B parameters.
  """
  NUM_STAGES = 1
  NUM_LAYERS = 16
  NUM_HEADS = 16
  MODEL_DIMS = 4096
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  PERCORE_BATCH_SIZE = 2  # 512 global batch size
  ICI_MESH_SHAPE = [1, 1, 2, 2]
  DCN_MESH_SHAPE = [1, 1, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  NUM_DECODE_BATCHES = 16
  DECODE_INTERVAL_STEPS = 50
  EVAL_INTERVAL_STEPS = 50
  PERCORE_EVAL_BATCH_SIZE = 1
  
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 100
    task_p.train.decode_interval_steps = 50

    return task_p
  
@experiment_registry.register
class CNNDMSpmdPipelineGpt3SmallAdam1X2X8(CNNDMSpmdPipelineAdam):
  """Small GPT-3 config in bf16 for 8 replicas with 512 global batch size.

  This was called GPT-3 XL in the GPT-3 paper, with 1.3B parameters.
  """
  NUM_STAGES = 1
  NUM_LAYERS = 16
  NUM_HEADS = 16
  MODEL_DIMS = 4096
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  PERCORE_BATCH_SIZE = 0.5  # 512 global batch size
  ICI_MESH_SHAPE = [1, 1, 4, 4]
  DCN_MESH_SHAPE = [1, 1, 1, 1]
  FPROP_DTYPE = jnp.bfloat16
  
  NUM_DECODE_BATCHES = 16
  DECODE_INTERVAL_STEPS = 50
  EVAL_INTERVAL_STEPS = 50
  PERCORE_EVAL_BATCH_SIZE = 1
  
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 100
    task_p.train.decode_interval_steps = 50

    return task_p
  
class GPTSpmdAdam(CNNDMSpmdAdam):
  r"""GPT-3 config with original HPs.

  From the paper & after convergence matching with
  NVIDIA's Megatron-LM framework.
  """
  MAX_SEQ_LEN = 2048

  NUM_LAYERS = 96
  NUM_HEADS = 96
  MODEL_DIMS = 12288
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 50257
  USE_REPEATED_LAYER = True

  # Model configs
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  ATTEN_LOGIT_CAP = -1.0  # Disable logits cap in atten

  # HPs
  LEARNING_RATE = 6e-5
  WEIGHT_DECAY = 0.1
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_CLIP_THRESHOLD = -1.0  # Disable Adam clip_threshold
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  LAYERNORM_EPSILON = 1e-5

  # In units of steps for BS1.5k
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 265
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 108600
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  ICI_MESH_SHAPE = [1, 1, 4, 4]
  DCN_MESH_SHAPE = [1, 1, 1, 1]
  
  # Training target
  TARGET_LOG_PPLX = 2.69

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10
  
  USE_REPEATED_LAYER = True
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT  # pytype: disable=attribute-error  # enable-nested-classes

    stacked_p = model_p.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
    if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if self.USE_REPEATED_LAYER:
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p

@experiment_registry.register
class GPTSpmdAdamMLPerfHP(GPTSpmdAdam):
  r"""GPT-3 config for MLPerf submission."""
  VOCAB_SIZE = 51200
  FPROP_DTYPE = jnp.bfloat16
  SUMMARY_INTERVAL_STEPS = 1000000
  # subclass must set the eval and the checkpoint intervals
  EVAL_INTERVAL_STEPS = None

  # Let set_adam_and_learning_rate_schedule calculate the following HPs
  # based on global batch size
  LEARNING_RATE = None
  LR_COS_WARMUP = None
  LR_COS_DECAY_START = None
  LR_COS_DECAY_END = None

  # PROFILER_CAPTURE_STEP = 2
  # PROFILER_MIN_DURATION_SEC = 100
  # PROFILER_MAX_NUM_HOSTS = 4

  # @property
  # def CHECKPOINT_EVERY_N_STEPS(self) -> int:
  #   value = super().CHECKPOINT_EVERY_N_STEPS
  #   if _MLPERF_GPT3_CHECKPOINT_EVERY_N_STEPS.value:
  #     value = int(_MLPERF_GPT3_CHECKPOINT_EVERY_N_STEPS.value)
  #   return value

  # @property
  # def CHECKPOINT_MAX_TO_KEEP(self) -> int:
  #   value = super().CHECKPOINT_MAX_TO_KEEP
  #   if _MLPERF_GPT3_CHECKPOINT_MAX_TO_KEEP.value:
  #     value = int(_MLPERF_GPT3_CHECKPOINT_MAX_TO_KEEP.value)
  #   return value

  # @property
  # def TRAINING_SEED(self) -> int:
  #   value = super().TRAINING_SEED
  #   if _MLPERF_GPT3_TRAINING_SEED.value:
  #     value = int(_MLPERF_GPT3_TRAINING_SEED.value)
  #   return value

  # @property
  # def TRAINING_NUM_BATCHES_TO_SKIP(self) -> int:
  #   value = super().TRAINING_NUM_BATCHES_TO_SKIP
  #   if _MLPERF_GPT3_TRAINING_NUM_BATCHES_TO_SKIP.value:
  #     value = int(_MLPERF_GPT3_TRAINING_NUM_BATCHES_TO_SKIP.value)
  #   return value

  # def task(self) -> tasks_lib.SingleTask.HParams:
  #   """Returns the task parameters."""
  #   mllogger = c4_mllog.try_to_init_mlloger(self)
  #   task_p = c4_mllog.log_params_and_set_early_stopping_fn(
  #       self, super().task(), mllogger
  #   )
  #   task_p.summary_verbosity = int(_MLPERF_GPT3_SUMMARY_VERBOSITY.value)
  #   return task_p
  

# class GPTSpmdAdamMLPerfHP(GPTSpmdAdamMLPerfHP):
#   r"""GPT-3 config for MLPerf submission."""
    
@experiment_registry.register
class CNNDMSpmdAdamCkpEval(GPTSpmdAdamMLPerfHP):
  """Small GPT-3 config in bf16 for 8 replicas with 512 global batch size.

  This was called GPT-3 XL in the GPT-3 paper, with 1.3B parameters.
  """
  PERCORE_BATCH_SIZE = 0.25  # 2048 global batch size
  ICI_MESH_SHAPE = [1, 8, 16]
  DCN_MESH_SHAPE = [1, 1, 1]
  FPROP_DTYPE = jnp.bfloat16
  
  SUMMARY_INTERVAL_STEPS = 100
  NUM_DECODE_BATCHES = 16
  DECODE_INTERVAL_STEPS = 50
  EVAL_INTERVAL_STEPS = 50
  PERCORE_EVAL_BATCH_SIZE = 0.25
  USE_REPEATED_LAYER = True
  
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    
    # Add a init rule to load from a pretrained model.
    task_p.train.init_from_checkpoint_rules = {
        'gs://test-example-123/GPT3/training_log/v4/175B/1/checkpoints/':
            tasks_lib.CheckpointLoadingRules(
                task_p=task_p,
                # Load all of TrainState.
                load_rules=[(r'(.*)', '{}')],
                load_step=True,
                load_opt_states=True,
                step=10_000,
                input_specs_provider_p=super().get_input_specs_provider_params(),
                )
    }
  
    task_p.train.num_train_steps = 1000
    task_p.train.decode_interval_steps = 50

    return task_p
     
@experiment_registry.register
class CNNDMSpmdRepeatedGpt3SmallAdam12X16X16(CNNDMSpmdAdamCkpEval):
  """Small GPT-3 config in bf16 for 8 replicas with 512 global batch size.

  This was called GPT-3 XL in the GPT-3 paper, with 1.3B parameters.
  """
  PERCORE_BATCH_SIZE = 2  # 2048 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [12, 1, 1]
  FPROP_DTYPE = jnp.bfloat16
  
  NUM_DECODE_BATCHES = 16
  DECODE_INTERVAL_STEPS = 50
  EVAL_INTERVAL_STEPS = 50
  PERCORE_EVAL_BATCH_SIZE = 4
  USE_REPEATED_LAYER = True
  
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 100
    task_p.train.decode_interval_steps = 50

    return task_p
  
@experiment_registry.register
class CNNDMSpmdRepeatedGpt3Adam1X8x16(GPTSpmdAdamMLPerfHP):
  """Small GPT-3 config in bf16 for 8 replicas with 512 global batch size.

  This was called GPT-3 XL in the GPT-3 paper, with 1.3B parameters.
  """
  PERCORE_BATCH_SIZE = 0.5  # 2048 global batch size
  ICI_MESH_SHAPE = [1, 128, 1]
  DCN_MESH_SHAPE = [1, 1, 1]
  FPROP_DTYPE = jnp.bfloat16
  
  NUM_DECODE_BATCHES = 16
  DECODE_INTERVAL_STEPS = 50
  EVAL_INTERVAL_STEPS = 50
  PERCORE_EVAL_BATCH_SIZE = 4
  USE_REPEATED_LAYER = True
  SUMMARY_INTERVAL_STEPS = 100
  
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 100
    task_p.train.decode_interval_steps = 50

    return task_p