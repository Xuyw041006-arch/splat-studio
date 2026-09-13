import pytest
from backend.semantic_worker import semantic_parameter_budget

GIB = 1024 ** 3

def test_a100_budget_admits_complete_fine_field_with_working_headroom():
    limit, audit = semantic_parameter_budget({}, 36 * GIB)
    assert limit == 27 * GIB
    assert 2162260 * 523 * 24 < limit
    assert audit['free_device_bytes'] - limit == 9 * GIB

def test_shared_gpu_available_memory_and_explicit_limits_are_respected():
    assert semantic_parameter_budget({}, 12 * GIB)[0] == 9 * GIB
    assert semantic_parameter_budget({'max_semantic_parameter_bytes': 3 * GIB}, 12 * GIB)[0] == 3 * GIB
    assert semantic_parameter_budget({'max_semantic_parameter_bytes': 30 * GIB}, 12 * GIB)[0] == 9 * GIB
    assert semantic_parameter_budget({}, 80 * GIB)[0] == 32 * GIB
    assert semantic_parameter_budget({}, None)[0] == 8 * GIB

@pytest.mark.parametrize('value', [True, 0, -1, 1.2, '8589934592'])
def test_invalid_explicit_memory_budget_is_rejected(value):
    with pytest.raises(ValueError):
        semantic_parameter_budget({'max_semantic_parameter_bytes': value}, 36 * GIB)

def test_empty_gpu_has_zero_allocatable_budget():
    assert semantic_parameter_budget({}, 0)[0] == 0
    assert semantic_parameter_budget({'max_semantic_parameter_bytes': 1}, 0)[0] == 0
