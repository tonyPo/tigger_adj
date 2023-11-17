
import pickle
import json
import enum
from pathlib import Path
from typing import Any, Callable, List, Dict, Type, Optional, Tuple, TypeVar, Union, cast, get_args, get_origin


class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value
    
def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')
    
    
def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')

def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))
    