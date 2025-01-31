import itertools
import random

SCRIPT_EXE = '$EXECUTABLE'
MID_COORDS_MACRO = '{mid_coordinates}'

def expand_macros(line):
    def read(coord: str):
        x = line.split(f'GRID_DIMENSIONS_{coord.upper()}="')[1].split('"')[0]
        return int(x)
    
    return line.replace(MID_COORDS_MACRO, str(read('x') // 2) + ',' + str(read('y') // 2))

class VariantGenerator:
    def __init__(self, prefix=''):
        self.prefix = prefix

    def generate(self, variants):
        for variant in itertools.product(*variants):
            yield variant

    def generate_to_arr_of_lines(self, variants):
        return [' '.join(variant) for variant in self.generate(variants)]

    def generate_to_str(self, variants):
        return '\n'.join(self.generate_to_arr_of_lines(variants))

class Generator:
    def __init__(self):
        self.algs_with_hps = []

    def set_algs_and_hps(self, algs_with_hps):
        self.algs_with_hps = algs_with_hps
        return self

    def generate_all(self):
        res = []
        for [alg, hps] in self.algs_with_hps:
            res.extend(self.generate_for_alg(alg))

        return res

    def generate_for_alg(self, alg_spec):
        hps = self._find(alg_spec)
        alg_variants = self._get_alg_variants(alg_spec)

        cases = [
            *hps,
            alg_variants,
        ]

        variant_lines = VariantGenerator().generate_to_arr_of_lines(cases)
        
        expanded = [expand_macros(line) for line in variant_lines]
        full_lines = [f'{line} {SCRIPT_EXE}' for line in expanded]

        return self.__shuffle_lines(full_lines)

    def _get_alg_variants(self, alg_spec):
        [alg_key, all_alg_variants] = alg_spec

        if all_alg_variants is None:
            all_alg_variants = ['']

        return [f' ALGORITHM="{alg_key}" {alg_variant} ' for alg_variant in all_alg_variants]

    def _find(self, alg_spec):
        for alg, hps in self.algs_with_hps:
            if alg == alg_spec:
                return hps

        return None
    
    def __shuffle_lines(self, lines):
        random.shuffle(lines)
        return lines

def interleaf_lines_with_echos(lines: list[str], start: int = 0):
    res = []
    for i, line in enumerate(lines.split('\n')):
        res.append(f'echo "exp-{i + start}"')
        res.append(line)
        res.append('\n')

    return '\n'.join(res)

def write_to_files(folder, lines, template_name, workers_count):
    parts = workers_count
    filenames = f'./{folder}/_scripts/{template_name}--part_{"{i}"}.sh'
    file_prefix = '#!/bin/bash\n\n'

    for i in range(parts):
        fname = filenames.replace('{i}', str(i + 1))
        print (f'Writing to {fname}')
        
        content = '\n'.join(lines[i::parts])
        content = interleaf_lines_with_echos(content, i * len(lines) // parts)
        
        with open(fname, 'w') as f:
            f.write(file_prefix)
            f.write(content)
