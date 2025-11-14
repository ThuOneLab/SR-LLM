from codes.dependencies.config.BaseConfig.baseConfig_update import get_DEFAULT_TRAIN_ARGS

DEFAULT_TRAIN_ARGS = get_DEFAULT_TRAIN_ARGS(
    formula="NEW")
DEFAULT_TRAIN_ARGS["bool_args"]["n_epochs"] = 50
DEFAULT_TRAIN_ARGS["bool_args"]["bool_use_evolutions"] = False