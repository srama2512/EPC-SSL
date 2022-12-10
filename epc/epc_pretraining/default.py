from yacs.config import CfgNode as CN

_C = CN()

# OPTIMIZATION SETTINGS
_C.LR = 1e-4
_C.SEED = 123
_C.EPOCHS = 1000
_C.BATCH_SIZE = 24
_C.WEIGHT_DECAY = 1e-6
_C.MAX_GRAD_NORM = 0.5
_C.NUM_WORKERS = 10
_C.PRETRAINED_PATH = ""
# LOGGING SETTINGS
_C.SAVE_DIR = "./"
_C.FLUSH_SECS = 30
# DATASET CONFIGURATIONS
_C.DATA_ROOT = "data/walkthrough_features/strong_exploration"
_C.IMAGE_HEIGHT = 256
_C.IMAGE_WIDTH = 352
_C.VIDEO_LENGTH = 499
_C.NUM_SAMPLES_PER_VIDEO = 6
# VISUAL ENCODER
_C.VISUAL_ENCODER = CN()
_C.VISUAL_ENCODER.encoder_type = "resnet"
_C.VISUAL_ENCODER.RESNET = CN()
_C.VISUAL_ENCODER.RESNET.encoder_type = "resnet"
_C.VISUAL_ENCODER.RESNET.ENCODER = CN()
_C.VISUAL_ENCODER.RESNET.ENCODER.input_shape = (256, 256, 4)
_C.VISUAL_ENCODER.RESNET.ENCODER.baseplanes = 32
_C.VISUAL_ENCODER.RESNET.ENCODER.backbone = "resnet18"
_C.VISUAL_ENCODER.RESNET.ENCODER.hidden_size = 128
_C.VISUAL_ENCODER.RESNET.ENCODER.normalize_visual_inputs = True
_C.VISUAL_ENCODER.RESNET.ENCODER.pretrained_path = (
    "pretrained_models/moco_encoder.pth.tar"
)
# TRANSFORMER CONFIGURATIONS
_C.TRANSFORMER = CN()
_C.TRANSFORMER.nhead = 8
_C.TRANSFORMER.num_encoder_layers = 1
_C.TRANSFORMER.num_decoder_layers = 1
_C.TRANSFORMER.dim_feedforward = 128
_C.TRANSFORMER.dropout = 0.0
_C.TRANSFORMER.activation = "relu"
# SELF_SUPERVISED_LEARNING CONFIGURATIONS
_C.SELF_SUPERVISED_LEARNING = CN()
_C.SELF_SUPERVISED_LEARNING.masking_mode = "temporal"
_C.SELF_SUPERVISED_LEARNING.random_segment_length = 5  # (start, end) range
_C.SELF_SUPERVISED_LEARNING.past_time_segment = 10
_C.SELF_SUPERVISED_LEARNING.future_time_segment = 10
# LOSS CONFIGURATIONS
_C.LOSS = CN()
_C.LOSS.ignore_query_pose = False  # use only time to query, not pose
# Contrastive loss configs
_C.LOSS.CONTRASTIVE = CN()
_C.LOSS.CONTRASTIVE.temperature = 0.1  # temperature value for l2 loss
_C.LOSS.CONTRASTIVE.normalization = "l2"  # can be none / l2 / bilinear
_C.LOSS.CONTRASTIVE.feature_dims = 128
# EVALUATION SETTINGS
_C.EVAL_INTERVAL = 100  # Evaluate after each epoch is completed
_C.EVAL_SAVE_INTERVAL = 100  # Save a unique checkpoint after every 5 epochs


def get_config(config_path=None, opts=None):
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    if config_path:
        config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)
    config.freeze()
    return config
