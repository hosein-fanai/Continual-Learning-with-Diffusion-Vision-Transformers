from tensorflow.keras import layers, models

from copy import deepcopy


class ArgumentSaver:
    """
    
    """

    def _save_init_args(
        self, 
        local_vars, 
        exclude=("self", "kwargs", "__class__", "temp_val"), 
        rename={"build": "build_"}, 
    ):
        if not hasattr(self, "_init_config"):
            self._init_config = {}

        for name, value in local_vars.items():
            if name in exclude:
                continue

            setattr(
                self, 
                rename.get(name, name), 
                value
            )

            self._init_config[name] = deepcopy(value) if isinstance(value, (list, set, dict)) else value

        return self._init_config

    def get_config(self):
        config = super().get_config()
        config.update(self._init_config)

        return config

    @classmethod
    def from_config(cls, config):
        config = deepcopy(config)

        return cls(**config)


class ArgumentSaverLayer(ArgumentSaver, layers.Layer):
    """
    
    """


class ArgumentSaverModel(ArgumentSaver, models.Model):
    """
    
    """
