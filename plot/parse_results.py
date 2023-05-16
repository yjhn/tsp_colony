from dataclasses import dataclass
from typing import Any, List, TypeVar, Type, cast, Callable


T = TypeVar("T")


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


@dataclass
class AlgorithmConstants:
    max_iterations: int
    colony_size: int
    exchange_generations: int
    nl_max: int
    p_cp: float
    p_rc: float
    p_l: float
    l_min: int
    l_max_mul: float
    r: float
    capital_l: float
    lowercase_q: int
    k: float

    @staticmethod
    def from_dict(obj: Any) -> 'AlgorithmConstants':
        assert isinstance(obj, dict)
        max_iterations = from_int(obj.get("max_iterations"))
        colony_size = from_int(obj.get("colony_size"))
        exchange_generations = from_int(obj.get("exchange_generations"))
        nl_max = from_int(obj.get("nl_max"))
        p_cp = from_float(obj.get("p_cp"))
        p_rc = from_float(obj.get("p_rc"))
        p_l = from_float(obj.get("p_l"))
        l_min = from_int(obj.get("l_min"))
        l_max_mul = from_float(obj.get("l_max_mul"))
        r = from_float(obj.get("r"))
        capital_l = from_float(obj.get("capital_l"))
        lowercase_q = from_float(obj.get("lowercase_q"))
        k = from_float(obj.get("k"))
        return AlgorithmConstants(max_iterations, colony_size, exchange_generations, nl_max, p_cp, p_rc, p_l, l_min, l_max_mul, r, capital_l, lowercase_q, k)

    def to_dict(self) -> dict:
        result: dict = {}
        result["max_iterations"] = from_int(self.max_iterations)
        result["colony_size"] = from_int(self.colony_size)
        result["exchange_generations"] = from_int(self.exchange_generations)
        result["nl_max"] = from_int(self.nl_max)
        result["p_cp"] = to_float(self.p_cp)
        result["p_rc"] = to_float(self.p_rc)
        result["p_l"] = to_float(self.p_l)
        result["l_min"] = from_int(self.l_min)
        result["l_max_mul"] = to_float(self.l_max_mul)
        result["r"] = from_float(self.r)
        result["capital_l"] = from_float(self.capital_l)
        result["lowercase_q"] = from_float(self.lowercase_q)
        result["k"] = from_float(self.k)
        return result


@dataclass
class Problem:
    name: str
    optimal_length: int

    @staticmethod
    def from_dict(obj: Any) -> 'Problem':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        optimal_length = from_int(obj.get("optimal_length"))
        return Problem(name, optimal_length)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["optimal_length"] = from_int(self.optimal_length)
        return result


@dataclass
class BenchConfig:
    process_count: int
    problem: Problem
    algorithm: str
    algorithm_constants: AlgorithmConstants
    benchmark_start_time_millis: int
    repeat_times: int

    @staticmethod
    def from_dict(obj: Any) -> 'BenchConfig':
        assert isinstance(obj, dict)
        process_count = from_int(obj.get("process_count"))
        problem = Problem.from_dict(obj.get("problem"))
        algorithm = from_str(obj.get("algorithm"))
        algorithm_constants = AlgorithmConstants.from_dict(obj.get("algorithm_constants"))
        benchmark_start_time_millis = from_int(obj.get("benchmark_start_time_millis"))
        repeat_times = from_int(obj.get("repeat_times"))
        return BenchConfig(process_count, problem, algorithm, algorithm_constants, benchmark_start_time_millis, repeat_times)

    def to_dict(self) -> dict:
        result: dict = {}
        result["process_count"] = from_int(self.process_count)
        result["problem"] = to_class(Problem, self.problem)
        result["algorithm"] = from_str(self.algorithm)
        result["algorithm_constants"] = to_class(AlgorithmConstants, self.algorithm_constants)
        result["benchmark_start_time_millis"] = from_int(self.benchmark_start_time_millis)
        result["repeat_times"] = from_int(self.repeat_times)
        return result


@dataclass
class RunResult:
    run_number: int
    found_optimal_tour: bool
    shortest_found_tour: int
    iteration_reached: int
    shortest_iteration_tours: List[List[int]]
    avg_iter_time_non_exch_micros: float
    avg_iter_time_exch_micros: float
    duration_millis: int

    @staticmethod
    def from_dict(obj: Any) -> 'RunResult':
        assert isinstance(obj, dict)
        run_number = from_int(obj.get("run_number"))
        found_optimal_tour = from_bool(obj.get("found_optimal_tour"))
        shortest_found_tour = from_int(obj.get("shortest_found_tour"))
        iteration_reached = from_int(obj.get("iteration_reached"))
        shortest_iteration_tours = from_list(lambda x: from_list(from_int, x), obj.get("shortest_iteration_tours"))
        avg_iter_time_non_exch_micros = from_float(obj.get("avg_iter_time_non_exch_micros"))
        avg_iter_time_exch_micros = from_float(obj.get("avg_iter_time_exch_micros"))
        duration_millis = from_int(obj.get("duration_millis"))
        return RunResult(run_number, found_optimal_tour, shortest_found_tour, iteration_reached, shortest_iteration_tours, avg_iter_time_non_exch_micros, avg_iter_time_exch_micros, duration_millis)

    def to_dict(self) -> dict:
        result: dict = {}
        result["run_number"] = from_int(self.run_number)
        result["found_optimal_tour"] = from_bool(self.found_optimal_tour)
        result["shortest_found_tour"] = from_int(self.shortest_found_tour)
        result["iteration_reached"] = from_int(self.iteration_reached)
        result["shortest_iteration_tours"] = from_list(lambda x: from_list(from_int, x), self.shortest_iteration_tours)
        result["avg_iter_time_non_exch_micros"] = to_float(self.avg_iter_time_non_exch_micros)
        result["avg_iter_time_exch_micros"] = to_float(self.avg_iter_time_exch_micros)
        result["duration_millis"] = from_int(self.duration_millis)
        return result


@dataclass
class BenchmarkData:
    bench_config: BenchConfig
    benchmark_duration_millis: int
    run_results: List[RunResult]

    @staticmethod
    def from_dict(obj: Any) -> 'BenchmarkData':
        assert isinstance(obj, dict)
        bench_config = BenchConfig.from_dict(obj.get("bench_config"))
        benchmark_duration_millis = from_int(obj.get("benchmark_duration_millis"))
        run_results = from_list(RunResult.from_dict, obj.get("run_results"))
        return BenchmarkData(bench_config, benchmark_duration_millis, run_results)

    def to_dict(self) -> dict:
        result: dict = {}
        result["bench_config"] = to_class(BenchConfig, self.bench_config)
        result["benchmark_duration_millis"] = from_int(self.benchmark_duration_millis)
        result["run_results"] = from_list(lambda x: to_class(RunResult, x), self.run_results)
        return result


def benchmark_data_from_dict(s: Any) -> BenchmarkData:
    return BenchmarkData.from_dict(s)


def benchmark_data_to_dict(x: BenchmarkData) -> Any:
    return to_class(BenchmarkData, x)


import json

def read_bench_data(path: str) -> BenchmarkData:
    with open(path, 'r') as file:
        data = json.load(file)
    return benchmark_data_from_dict(data)

data = read_bench_data("results/bm_qCABC_kroA100_2cpus_cs40_nlmax5_pcp0.8_prc0.5_pl0.2_lmin2_lmaxm0.5_r1_q3_e32_k16_cl3.json")
print(data.to_dict())
