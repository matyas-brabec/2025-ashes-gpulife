import os
from typing import Callable

EXPERIMENTS_SEPARATOR = 'next-experiment'
MEASUREMENTS_SEPARATOR = 'Time report:'

class Key:
    algorithm_name='algorithm_name'
    grid_dimensions='grid_dimensions'
    grid_dim_x='<special_case_grid_dim_x>'
    grid_dim_y='<special_case_grid_dim_y>'
    iterations='iterations'
    base_grid_encoding='base_grid_encoding'
    max_runtime_seconds='max_runtime_seconds'
    warmup_rounds='warmup_rounds'
    measurement_rounds='measurement_rounds'
    data_loader_name='data_loader_name'
    pattern_expression='pattern_expression'
    measure_speedup='measure_speedup'
    speedup_bench_algorithm_name='speedup_bench_algorithm_name'
    validate='validate'
    print_validation_diff='print_validation_diff'
    validation_algorithm_name='validation_algorithm_name'
    animate_output='animate_output'
    colorful='colorful'
    random_seed='random_seed'
    state_bits_count='state_bits_count'
    thread_block_size='thread_block_size'
    warp_dims_x='warp_dims_x'
    warp_dims_y='warp_dims_y'
    warp_tile_dims_x='warp_tile_dims_x'
    warp_tile_dims_y='warp_tile_dims_y'
    streaming_direction='streaming_direction'
    tag='tag'
    
class MeasurementKey:
    set_and_format_input_data='set_and_format_input_data'
    initialize_data_structures='initialize_data_structures'
    run='run'
    performed_iters='performed iters'
    runtime_per_iter='runtime per iter'
    finalize_data_structures='finalize_data_structures'

class Measurement:
    def __init__(self, content: str, experiment: 'Experiment'):
        self.content: str = content.strip()
        self.experiment: 'Experiment' = experiment

    def get_value(self, key: str) -> float:
        raw = self._load_raw(key)
        try:
            return float(raw)
        except:
            return raw
    
    def _load_raw(self, key: str):
        splitted_by_key = self.content.split(f'{key}:')

        if (len(splitted_by_key) < 2):
            return None

        if (key == MeasurementKey.performed_iters):
            return splitted_by_key[1].split('\n')[0].strip()
        else:
            return splitted_by_key[1].split('ms')[0].strip()

    def compute_runtime_per_iter(self):
        iterations = self.get_value(MeasurementKey.performed_iters)
        runtime = self.get_value(MeasurementKey.run)

        if None not in [iterations, runtime]:
            return runtime / iterations

    def compute_runtime_per_cell_per_iter(self):
        runtime_per_iter = self.compute_runtime_per_iter()
        cell_count = self.experiment.compute_grid_size()
        
        if None not in [runtime_per_iter, cell_count]:
            return runtime_per_iter / cell_count

class MeasurementSet:
    def __init__(self, measurements: list[Measurement]):    
        self.measurements: list[Measurement] = measurements

    def get_median(self, prop_accessor: Callable[[Measurement], float]) -> float:
        
        values = self.get_valid_vals(prop_accessor) 
        values.sort()

        if len(values) == 0:
            return None
        
        if len(values) == 1:
            return values[0]
        
        return values[(len(values) + 1) // 2]

    def get_mean(self, prop_accessor: Callable[[Measurement], float]) -> float:
        values = self.get_valid_vals(prop_accessor)
        return sum(values) / len(values)

    def get_min(self, prop_accessor: Callable[[Measurement], float]) -> float:
        values = self.get_valid_vals(prop_accessor)
        return min(values)
    
    def get_max(self, prop_accessor: Callable[[Measurement], float]) -> float:
        values = self.get_valid_vals(prop_accessor)
        return max(values)
    
    def get_variance(self, prop_accessor: Callable[[Measurement], float]) -> float:
        values = self.get_valid_vals(prop_accessor)
        mean = self.get_mean(prop_accessor)
        return sum([(v - mean) ** 2 for v in values]) / len(values)
    
    def get_valid_vals(self, prop_accessor: Callable[[Measurement], float]) -> list[float]:
        vals = [prop_accessor(m) for m in self.measurements]
        return [v for v in vals if v is not None]

class Experiment:
    def __init__(self, content: str):
        self.content: str = content.strip()

    def get_param(self, key: str):
        if key in [Key.grid_dim_x, Key.grid_dim_y]:
            raw = self._load_raw(Key.grid_dimensions)
            return self._parse_dim(key, raw)
        
        raw = self._load_raw(key)
        
        try:
            return int(raw)
        except:
            return raw
        
    def _parse_dim(self, dim_idx: str, raw_dims: str):
        both_dims = raw_dims.split('x')

        if dim_idx == Key.grid_dim_x:
            return int(both_dims[0])
        elif dim_idx == Key.grid_dim_y:
            return int(both_dims[1])
        
    def _load_raw(self, key: str):
        try:
            return self.content.split(f'{key}:')[1].split('\n')[0].strip()
        except:
            return None
    
    def get_measurements(self) -> list[Measurement]:
        measurements = []
        for measurement_content in self.content.split(MEASUREMENTS_SEPARATOR)[1:]:
            measurements.append(Measurement(measurement_content , self))
        return measurements
    
    def get_measurement_set(self) -> MeasurementSet:
        return MeasurementSet(self.get_measurements())
        
    def matches(self, key_vals: list[tuple[str, any]]):
        for key, val in key_vals:
            if self.get_param(key) != val:
                return False
        return True
    
    def get_median_runtime_per_iter(self) -> float:
        measurements = self.get_measurements()
        runtimes = [m.get_value(MeasurementKey.runtime_per_iter) for m in measurements]

        if len(runtimes) == 0:
            return None
        
        if len(runtimes) == 1:
            return runtimes[0]

        runtimes.sort()
        return runtimes[(len(runtimes) + 1) // 2]

    def __str__(self):
        return f'Experiment: {self.get_param(Key.algorithm_name)} on grid {self.get_param(Key.grid_dimensions)} ...'

    def compute_grid_size(self):
        x = self.get_param(Key.grid_dim_x)
        y = self.get_param(Key.grid_dim_y)
        return x * y

class Results:
    def __init__(self, results_content: str):
        self.experiments: list[Experiment] = []

        for exp_content in results_content.split(EXPERIMENTS_SEPARATOR)[1:]:
            self.experiments.append(Experiment(exp_content))

    @staticmethod
    def from_file(file_name) -> 'Results':
        file_contents = None
        with open(file_name, 'r') as f:
            file_contents = f.read()

        return Results(file_contents)

    @staticmethod
    def from_directory(dir_name) -> 'Results':
        results = None

        for file_name in os.listdir(dir_name):
            file_path = os.path.join(dir_name, file_name)
            if os.path.isfile(file_path):
                f_results = Results.from_file(file_path)

                if results is None:
                    results = f_results
                else:
                    results.extend_with(f_results)

        return results

    def get_experiments_with(self, key_vals: list[tuple[str, any]]) -> list[Experiment]:
        return [exp for exp in self.experiments if exp.matches(key_vals)]

    def extend_with(self, results: 'Results'):
        self.experiments.extend(results.experiments)