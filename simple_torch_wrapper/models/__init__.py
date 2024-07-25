# simpleTorchWrapper/simple_torch_wrapper/models/__init__.py

import importlib
import sys
import os

class LazyLoader:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def _load(self):
        if self.module is None:
            self.module = importlib.import_module(f'.{self.module_name}', package='simple_torch_wrapper.models')
        return self.module

    def __getattr__(self, item):
        return getattr(self._load(), item)

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

# Initialize the __all__ variable
__all__ = []

# Loop through all the files in the directory
for filename in os.listdir(current_dir):
    # Check if the file is a Python file but not __init__.py
    if filename.endswith('.py') and filename != '__init__.py':
        # Get the module name by removing the file extension
        module_name = filename[:-3]
        # Add the module name to __all__
        __all__.append(module_name)
        # Add the lazy loader to sys.modules
        sys.modules[f'simple_torch_wrapper.models.{module_name}'] = LazyLoader(module_name)
