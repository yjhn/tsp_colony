from dataclasses import dataclass
from typing import Optional, Any, List, TypeVar, Type, cast, Callable
import json

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


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


# struct AntCycleConstants {
#     max_iterations: u32,
#     population_size: u32,
#     exchange_generations: u32,
#     alpha: Float,
#     beta: Float,
#     capital_q_mul: Float,
#     ro: Float,
#     lowercase_q: usize,
#     // init_g: u32,
#     k: u32,
#     init_intensity: Float,
# }
# struct QcabcConstants {
#     max_iterations: u32,
#     colony_size: u32,
#     exchange_generations: u32,
#     nl_max: u16,
#     p_cp: Float,
#     p_rc: Float,
#     p_l: Float,
#     l_min: usize,
#     l_max_mul: Float,
#     r: Float,
#     capital_l: Float,
#     lowercase_q: usize,
#     // initial_g: u32,
#     k: Float,
# }
@dataclass(frozen=True, kw_only=True)
class AlgorithmConstants:
    max_iterations: int
    exchange_generations: int
    lowercase_q: int
    k: float
    # [q]CABC-specific
    colony_size: Optional[int] = None  # CABC
    nl_max: Optional[int] = None  # CABC
    p_cp: Optional[float] = None  # CABC
    p_rc: Optional[float] = None  # CABC
    p_l: Optional[float] = None  # CABC
    l_min: Optional[int] = None  # CABC
    l_max_mul: Optional[float] = None  # CABC
    r: Optional[float] = None  # CABC
    capital_l: Optional[float] = None  # CABC
    # PACO-specific
    population_size: Optional[int] = None  # PACO
    init_intensity: Optional[float] = None  # PACO
    alpha: Optional[float] = None  # PACO
    beta: Optional[float] = None  # PACO
    capital_q_mul: Optional[float] = None  # PACO
    ro: Optional[float] = None  # PACO

    @staticmethod
    def from_dict(obj: Any) -> 'AlgorithmConstants':
        assert isinstance(obj, dict)
        max_iterations = from_int(obj.get("max_iterations"))
        colony_size = from_union([from_int, from_none], obj.get("colony_size"))
        population_size = from_union([from_int, from_none],
                                     obj.get("population_size"))
        exchange_generations = from_int(obj.get("exchange_generations"))
        nl_max = from_union([from_int, from_none], obj.get("nl_max"))
        p_cp = from_union([from_float, from_none], obj.get("p_cp"))
        p_rc = from_union([from_float, from_none], obj.get("p_rc"))
        p_l = from_union([from_float, from_none], obj.get("p_l"))
        l_min = from_union([from_int, from_none], obj.get("l_min"))
        l_max_mul = from_union([from_float, from_none], obj.get("l_max_mul"))
        r = from_union([from_float, from_none], obj.get("r"))
        capital_l = from_union([from_float, from_none], obj.get("capital_l"))
        lowercase_q = from_int(obj.get("lowercase_q"))
        k = from_float(obj.get("k"))
        init_intensity = from_union([from_float, from_none],
                                    obj.get("init_intensity"))
        alpha = from_union([from_float, from_none], obj.get("alpha"))
        beta = from_union([from_float, from_none], obj.get("beta"))
        capital_q_mul = from_union([from_float, from_none],
                                   obj.get("capital_q_mul"))
        ro = from_union([from_float, from_none], obj.get("ro"))
        return AlgorithmConstants(max_iterations=max_iterations,
                                  colony_size=colony_size,
                                  population_size=population_size,
                                  exchange_generations=exchange_generations,
                                  nl_max=nl_max,
                                  p_cp=p_cp,
                                  p_rc=p_rc,
                                  p_l=p_l,
                                  l_min=l_min,
                                  l_max_mul=l_max_mul,
                                  r=r,
                                  capital_l=capital_l,
                                  lowercase_q=lowercase_q,
                                  k=k,
                                  init_intensity=init_intensity,
                                  alpha=alpha,
                                  beta=beta,
                                  capital_q_mul=capital_q_mul,
                                  ro=ro)

    def to_dict(self) -> dict:
        result: dict = {}
        result["max_iterations"] = from_int(self.max_iterations)
        result["exchange_generations"] = from_int(self.exchange_generations)
        result["lowercase_q"] = from_int(self.lowercase_q)
        result["k"] = to_float(self.k)
        if self.colony_size is not None:
            # [q]CABC
            result["colony_size"] = from_int(self.colony_size)
            result["nl_max"] = from_int(self.nl_max)
            result["p_cp"] = to_float(self.p_cp)
            result["p_rc"] = from_union([to_float, from_none], self.p_rc)
            result["p_l"] = from_union([to_float, from_none], self.p_l)
            result["l_min"] = from_union([from_int, from_none], self.l_min)
            result["l_max_mul"] = from_union([to_float, from_none],
                                             self.l_max_mul)
            result["capital_l"] = from_union([to_float, from_none],
                                             self.capital_l)
        elif self.population_size is not None:
            # PACO
            result["population_size"] = from_int(self.population_size)
            result["init_intensity"] = from_union([to_float, from_none],
                                                  self.init_intensity)
            result["alpha"] = from_union([to_float, from_none], self.alpha)
            result["beta"] = from_union([to_float, from_none], self.beta)
            result["capital_q_mul"] = from_union([to_float, from_none],
                                                 self.capital_q_mul)
            result["ro"] = from_union([to_float, from_none], self.ro)
            result["r"] = from_union([to_float, from_none], self.r)
        return result


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
        algorithm_constants = AlgorithmConstants.from_dict(
            obj.get("algorithm_constants"))
        benchmark_start_time_millis = from_int(
            obj.get("benchmark_start_time_millis"))
        repeat_times = from_int(obj.get("repeat_times"))
        return BenchConfig(process_count, problem, algorithm,
                           algorithm_constants, benchmark_start_time_millis,
                           repeat_times)

    def to_dict(self) -> dict:
        result: dict = {}
        result["process_count"] = from_int(self.process_count)
        result["problem"] = to_class(Problem, self.problem)
        result["algorithm"] = from_str(self.algorithm)
        result["algorithm_constants"] = to_class(AlgorithmConstants,
                                                 self.algorithm_constants)
        result["benchmark_start_time_millis"] = from_int(
            self.benchmark_start_time_millis)
        result["repeat_times"] = from_int(self.repeat_times)
        return result


@dataclass(frozen=True)
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
        shortest_iteration_tours = from_list(
            lambda x: from_list(from_int, x),
            obj.get("shortest_iteration_tours"))
        avg_iter_time_non_exch_micros = from_float(
            obj.get("avg_iter_time_non_exch_micros"))
        avg_iter_time_exch_micros = from_float(
            obj.get("avg_iter_time_exch_micros"))
        duration_millis = from_int(obj.get("duration_millis"))
        return RunResult(run_number, found_optimal_tour, shortest_found_tour,
                         iteration_reached, shortest_iteration_tours,
                         avg_iter_time_non_exch_micros,
                         avg_iter_time_exch_micros, duration_millis)

    def to_dict(self) -> dict:
        result: dict = {}
        result["run_number"] = from_int(self.run_number)
        result["found_optimal_tour"] = from_bool(self.found_optimal_tour)
        result["shortest_found_tour"] = from_int(self.shortest_found_tour)
        result["iteration_reached"] = from_int(self.iteration_reached)
        result["shortest_iteration_tours"] = from_list(
            lambda x: from_list(from_int, x), self.shortest_iteration_tours)
        result["avg_iter_time_non_exch_micros"] = to_float(
            self.avg_iter_time_non_exch_micros)
        result["avg_iter_time_exch_micros"] = to_float(
            self.avg_iter_time_exch_micros)
        result["duration_millis"] = from_int(self.duration_millis)
        return result


@dataclass(frozen=True)
class BenchmarkData:
    bench_config: BenchConfig
    benchmark_duration_millis: int
    run_results: List[RunResult]

    @staticmethod
    def from_dict(obj: Any) -> 'BenchmarkData':
        assert isinstance(obj, dict)
        bench_config = BenchConfig.from_dict(obj.get("bench_config"))
        benchmark_duration_millis = from_int(
            obj.get("benchmark_duration_millis"))
        run_results = from_list(RunResult.from_dict, obj.get("run_results"))
        return BenchmarkData(bench_config, benchmark_duration_millis,
                             run_results)

    def to_dict(self) -> dict:
        result: dict = {}
        result["bench_config"] = to_class(BenchConfig, self.bench_config)
        result["benchmark_duration_millis"] = from_int(
            self.benchmark_duration_millis)
        result["run_results"] = from_list(lambda x: to_class(RunResult, x),
                                          self.run_results)
        return result


def benchmark_data_from_dict(s: Any) -> BenchmarkData:
    return BenchmarkData.from_dict(s)


def benchmark_data_to_dict(x: BenchmarkData) -> Any:
    return to_class(BenchmarkData, x)


def read_bench_data(path: str) -> BenchmarkData:
    with open(path, 'r') as file:
        data = json.load(file)
    return benchmark_data_from_dict(data)
