# simpleTorchWrapper/simple_torch_wrapper/models/__init__.py

import importlib
import sys

class LazyModule:
    def __init__(self, module_name, package):
        # full module path
        self.module_name = module_name
        # the package which the module resides 
        self.package = package
        self.module = None

    def _load(self):
        if self.module is None:
            self.module = importlib.import_module(self.module_name, self.package)
        return self.module

    def __getattr__(self, attr):
        return getattr(self._load(), attr)

# Manually add the models for lazy loading
transfer_learning_torchvision = LazyModule('simple_torch_wrapper.models.transfer_learning_torchvision', 'simple_torch_wrapper.models')
vit_regressor = LazyModule('simple_torch_wrapper.models.vit_regressor', 'simple_torch_wrapper.models')

# Add the lazy loaders to sys.modules
sys.modules['simple_torch_wrapper.models.transfer_learning_torchvision'] = transfer_learning_torchvision
sys.modules['simple_torch_wrapper.models.vit_regressor'] = vit_regressor

# Also set the modules as attributes of models for direct access
setattr(sys.modules[__name__], 'transfer_learning_torchvision', transfer_learning_torchvision)
setattr(sys.modules[__name__], 'vit_regressor', vit_regressor)

# Define __all__ for explicit imports
__all__ = ['transfer_learning_torchvision', 'vit_regressor']
