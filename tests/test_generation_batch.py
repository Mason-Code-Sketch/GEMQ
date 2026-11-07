"""Request-length generation probes preserve the native generation path."""
from types import SimpleNamespace
import pytest
from gemq.generation_batch import install_generation_batch_probe


class Harness:
    batch_size = "auto"
    max_batch_size = 64
    max_length = 40960
    backend = "causal"
    world_size = 1

    def __init__(self):
        self.calls = []
        self.formal_requests = None
        self.fail = False

    def tok_encode(self, text):
        return list(range(int(text)))

    def _detect_batch_size(self, requests=None, pos=0):
        self.calls.append(requests)
        return 57

    def generate_until(self, requests, disable_tqdm=False):
        self.formal_requests = requests
        if self.batch_size == "auto":
            self._detect_batch_size()
        assert self.max_length == 40960 or self.max_length == 300
        if self.fail:
            raise RuntimeError("generation failed")
        return [r.args[0] for r in requests]


@pytest.fixture(autouse=True)
def mock_device_probe(monkeypatch):
    def probe(wrapped, tokens, context_tokens=None):
        wrapped.calls.append([(None, [1] * tokens, [1])])
        return 57
    monkeypatch.setattr("gemq.generation_batch.probe_generation_batch", probe)


def request(length, budget=256):
    return SimpleNamespace(args=(str(length), {"max_gen_toks": budget}))


def test_actual_length_plus_generation_budget_and_unchanged_requests():
    wrapped = Harness()
    records = install_generation_batch_probe(wrapped)
    requests = [request(192), request(20)]
    assert wrapped.generate_until(requests) == ["192", "20"]
    assert wrapped.formal_requests is requests
    assert len(wrapped.calls[0][0][1]) == 448
    assert wrapped.max_length == 40960
    assert "_detect_batch_size" not in wrapped.__dict__
    assert records == [{"max_input_tokens": 192, "max_new_tokens": 256,
                        "probe_tokens": 448, "model_max_length": 40960,
                        "max_batch_size": 64, "selected_batch_size": 57}]


def test_context_limit_and_per_request_budget():
    wrapped = Harness(); wrapped.max_length = 300
    records = install_generation_batch_probe(wrapped)
    requests = [request(200, 256), request(100, 8)]
    assert wrapped.generate_until(requests) == ["200", "100"]
    assert len(wrapped.calls[0][0][1]) == 300
    assert records[0]["probe_tokens"] == 300
    assert wrapped.max_length == 300


def test_later_calls_recompute_length_and_restore_on_error():
    wrapped = Harness()
    records = install_generation_batch_probe(wrapped)
    wrapped.generate_until([request(192)])
    wrapped.fail = True
    with pytest.raises(RuntimeError, match="generation failed"):
        wrapped.generate_until([request(30)])
    assert [len(call[0][1]) for call in wrapped.calls] == [448, 286]
    assert "_detect_batch_size" not in wrapped.__dict__
    assert len(records) == 2


def test_existing_instance_detector_is_restored():
    wrapped = Harness()
    original = lambda requests=None, pos=0: 32
    wrapped._detect_batch_size = original
    records = install_generation_batch_probe(wrapped)
    wrapped.generate_until([request(20)])
    assert wrapped._detect_batch_size is original
    assert records[0]["selected_batch_size"] == 57


def test_fixed_batch_and_empty_requests_skip_probe():
    wrapped = Harness(); wrapped.batch_size = 4
    records = install_generation_batch_probe(wrapped)
    assert wrapped.generate_until([request(192)]) == ["192"]
    assert wrapped.calls == [] and records == []
    wrapped.batch_size = "auto"
    assert wrapped.generate_until([]) == []
    assert wrapped.calls == []


@pytest.mark.parametrize("budget", [0, -1, 40960])
def test_invalid_budget_fails_before_mutating_detector(budget):
    wrapped = Harness()
    install_generation_batch_probe(wrapped)
    with pytest.raises(ValueError, match="generation budget"):
        wrapped.generate_until([request(10, budget)])
    assert "_detect_batch_size" not in wrapped.__dict__


def test_distributed_evaluator_uses_native_rank_synchronization():
    wrapped = Harness(); wrapped.world_size = 2
    records = install_generation_batch_probe(wrapped)
    wrapped.generate_until([request(20)])
    assert len(wrapped.calls[0][0][1]) == 276
    assert records[0]["selected_batch_size"] == 57
