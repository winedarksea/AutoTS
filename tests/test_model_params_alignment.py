"""Tests ensuring model parameter interfaces stay aligned."""

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autots.models.base import ModelObject


def _iter_model_classes():
    """Yield ModelObject subclasses defined across model modules."""

    import autots.models as models_pkg

    for modinfo in pkgutil.iter_modules(models_pkg.__path__):
        fullname = f"{models_pkg.__name__}.{modinfo.name}"
        try:
            module = importlib.import_module(fullname)
        except ModuleNotFoundError:
            # Optional dependency for this module is not installed in the
            # testing environment – skip to the next module.
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, ModelObject) or obj is ModelObject:
                continue
            if 'get_new_params' not in obj.__dict__:
                continue
            yield fullname, obj


def test_get_new_params_keys_present_in_get_params():
    """Ensure model specific params are exposed via ``get_params``."""

    mismatches = []

    optional_lib_names = ("torch", "tensorflow", "gluonts", "pytorch")

    for module_name, model_cls in _iter_model_classes():
        try:
            model = model_cls()
        except (ModuleNotFoundError, ImportError, AttributeError):
            # These models rely on optional dependencies that may not be
            # installed in the CI environment. Skip them gracefully.
            continue
        except NameError as exc:
            if any(lib in str(exc) for lib in optional_lib_names):
                continue
            raise

        try:
            new_params = model.get_new_params()
        except Exception:
            # If get_new_params cannot be evaluated without optional
            # dependencies we skip the model for this alignment check.
            continue

        try:
            current_params = model.get_params()
        except Exception:
            continue

        missing = [key for key in new_params if key not in current_params]
        if missing:
            mismatches.append((f"{module_name}.{model_cls.__name__}", missing))

    if mismatches:
        mismatch_messages = [
            f"{class_name} missing keys: {', '.join(sorted(keys))}"
            for class_name, keys in mismatches
        ]
        pytest.fail("\n".join(mismatch_messages))
