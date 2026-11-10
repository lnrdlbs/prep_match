import dataclasses
import uuid
import numpy as _np
import inspect as _inspect
import json as _json

class SmartEncoder(_json.JSONEncoder):

    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, uuid.UUID):
            return str(obj)

        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (_np.integer, _np.floating, _np.bool_)):
            return obj.item()

        if hasattr(obj, "__dict__"):
            return {
                k: v
                for k, v in vars(obj).items()
                if not (k.startswith("__") and k.endswith("__"))
                and not _inspect.isroutine(v)
            }

        return super().default(obj)