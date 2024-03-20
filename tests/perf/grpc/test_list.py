import pytest
import cProfile, pstats, io
from pstats import SortKey

def benchmark_config():
    return {
        'iterations': 10, 
        'rounds': 10
    }

def test_list_paginated(benchmark, idx):
    benchmark.pedantic(idx.list_paginated, **benchmark_config())

def test_list_paginated2(benchmark, idx):
        benchmark.pedantic(idx.list_paginated2, **benchmark_config())

def test_list_cProfile(idx):
    pr = cProfile.Profile()
    pr.enable()
    idx.list_paginated2()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
