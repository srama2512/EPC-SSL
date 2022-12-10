from yacs.config import CfgNode as CN

_C = CN()

# ==================================== Model config ===================================
_C.MODEL = CN()
# ENCODER config
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.input_shape = (256, 256, 4)
_C.MODEL.ENCODER.baseplanes = 32
_C.MODEL.ENCODER.backbone = "resnet18"
_C.MODEL.ENCODER.hidden_size = 128
_C.MODEL.ENCODER.normalize_visual_inputs = True
# PROJECTOR config
_C.MODEL.PROJECTOR = CN()
_C.MODEL.PROJECTOR.dim = 128


def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
