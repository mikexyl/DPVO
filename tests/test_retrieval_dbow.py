import importlib.util
import sys
import threading
import types
from pathlib import Path

import numpy as np


class _FakeDPRetrieval:
    def __init__(self, _vocabulary_path, _radius):
        pass

    def insert_image(self, _image):
        return np.ones(1, dtype=np.float32)

    def query(self, _index):
        return 0.0, 0, None


def _load_retrieval_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "dpretrieval",
        types.SimpleNamespace(DPRetrieval=_FakeDPRetrieval),
    )
    monkeypatch.setitem(
        sys.modules,
        "einops",
        types.SimpleNamespace(parse_shape=lambda *_args, **_kwargs: {"RGB": 3}),
    )
    module_path = (
        Path(__file__).parents[1]
        / "dpvo"
        / "loop_closure"
        / "retrieval"
        / "retrieval_dbow.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_retrieval_dbow_module", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_flush_cannot_deadlock_on_full_result_queue(tmp_path, monkeypatch):
    module = _load_retrieval_module(monkeypatch)
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.touch()
    retrieval = module.RetrievalDBOW(str(vocabulary), use_threads=True)

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    retrieval.image_buffer.update({index: image for index in range(64)})

    producer = threading.Thread(target=retrieval.save_up_to, args=(63,), daemon=True)
    producer.start()
    producer.join(timeout=2.0)

    assert not producer.is_alive(), "final DBoW flush deadlocked"
    assert retrieval.out_queue.maxsize == 0
    retrieval.detect_loop(thresh=1.0)
    retrieval.close()
