class Site:
    LEFT = 'left'
    RIGHT = 'right'
    TOP = 'top'
    BOTTOM = 'bottom'

class VAR:
    lt = 'LT'
    ct = 'CT'
    rt = 'RT'

    lc = 'LC'
    cc = 'CC'
    rc = 'RC'

    lb = 'LB'
    cb = 'CB'
    rb = 'RB'

class Num:
    cpp_16 = 'std::uint16_t'
    cpp_32 = 'std::uint32_t'
    cpp_64 = 'std::uint64_t'

    def __init__(self, bits):
        self.bit_count = bits

        
        if bits == 64:
            self.x = 8
            self.y = 8
        elif bits == 32:
            self.x = 8
            self.y = 4
        elif bits == 16:
            self.x = 4
            self.y = 4
        else:
            raise ValueError('Only 16, 32 and 64 bits are supported')

        self.bits = [
            [0] * self.x for _ in range(self.y)
        ]

    @staticmethod
    def zero(bits):
        return Num(bits)

    def __str__(self):
        return self.to_const()
    
    def set(self, x, y, value=1):
        self.__assert_bounds(x, y)

        self.bits[y][x] = value
        return self
    
    def to_grid_view(self):
        res = ''
        for bit_line in self.bits:
            for bit in bit_line:
                res += str(bit)
            res += '\n'
        return res


    def to_const(self):
        str_rep = self.to_grid_view().replace('\n', '').lstrip('0')
        constexpr = f'0b{str_rep if len(str_rep) > 0 else 0}' 
        cpp_type = self._get_cpp_type()

        return f'static_cast<{cpp_type}>({constexpr})'

    def _get_cpp_type(self):
        if self.bit_count == 32:
            return Num.cpp_32
        elif self.bit_count == 64:
            return Num.cpp_64
        elif self.bit_count == 16:
            return Num.cpp_16
        
    def set_neighborhood_of(self, x, y, value=1):
        self.__assert_bounds(x, y)
        
        self._set_neighborhood_in_ranges(
            self._x_range(x, y),
            self._y_range(x, y),
            x, y,
            value
        )
        
        return self

    def __assert_bounds(self, x, y):
        if x < 0 or x >= self.x or y < 0 or y >= self.y:
            raise ValueError(f'Index out of bounds: ({x}, {y})')

    def set_neighborhood_being(self, *sites):
        num = self

        if len(sites) > 1:
            return self._set_conner_neighborhood(sites)

        class SiteConnerImpl:
            def neighbor_at(self, x, y):
                return num._set_neighborhood_being(sites[0], x, y)
        return SiteConnerImpl()

    def _set_neighborhood_being(self, site, x, y):
        self.__assert_bounds(x, y)

        if site == Site.BOTTOM:
            x_range = self._x_range(x, y)
            y_range = range(0, 1)

        elif site == Site.TOP:
            x_range = self._x_range(x, y)
            y_range = range(self.y - 1, self.y)

        elif site == Site.RIGHT:
            x_range = range(0, 1)
            y_range = self._y_range(x, y)

        elif site == Site.LEFT:
            x_range = range(self.x - 1, self.x)
            y_range = self._y_range(x, y)

        self._set_neighborhood_in_ranges(x_range, y_range)

        return self

    def _set_conner_neighborhood(self, conner_spec):
        _k = self._conner_key

        if _k(conner_spec) == _k([Site.TOP, Site.LEFT]):
            self.set(self.x - 1, self.y - 1)
        
        if _k(conner_spec) == _k([Site.TOP, Site.RIGHT]):
            self.set(0, self.y - 1)
        
        if _k(conner_spec) == _k([Site.BOTTOM, Site.LEFT]):
            self.set(self.x - 1, 0)
        
        if _k(conner_spec) == _k([Site.BOTTOM, Site.RIGHT]):
            self.set(0, 0)
        
        return self

    def _set_neighborhood_in_ranges(self,
            x_range, y_range,
            x_origin=None, y_origin=None,
            value=1):

        for x in x_range:
            for y in y_range:
                if x == x_origin and y == y_origin:
                    continue
                self.set(x, y, value)
            
    def _x_range(self, x, y):
        x_start = max(0, x - 1)
        x_end = min(self.x, x + 2)
        
        return range(x_start, x_end)

    def _y_range(self, x, y):
        y_start = max(0, y - 1)
        y_end = min(self.y, y + 2)

        return range(y_start, y_end)

    def _conner_key(self, conner_spec):
        conner_spec = [*conner_spec]
        conner_spec.sort()
        return '-'.join(conner_spec)
    
    def set_neighborhood_mod(self, x, y, value=1):
        self.__assert_bounds(x, y)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                
                self.set(self._x_mod(x + dx), self._y_mod(y + dy), value)

        return self
    
    def _x_mod(self, x):
        return (x + self.x) % self.x

    def _y_mod(self, y):
        return (y + self.y) % self.y

    def max_order_of_set_bit(self):
        str_rep = self.to_grid_view().replace('\n', '').lstrip('0')
        return len(str_rep) - 1

    @staticmethod
    def cast_expr(expr, to_bits):
        return f'static_cast<{Num(to_bits)._get_cpp_type()}>({expr})'

class GOL:
    def __init__(self, bits):
        self.bits = bits
        self._debug_mask = Num.zero(bits)

    def get_macro(self, vars):
        inner_cells = self._compute_inner_cells(vars.cc)
        site_cells = self._compute_site_cells(vars)
        conner_cells = self._compute_conner_cells(vars)

        return GOL._or(inner_cells, site_cells, conner_cells)
    
    def _compute_inner_cells(self, cc):
        cell_results = []

        for x in range(1, Num(self.bits).x - 1):
            for y in range(1, Num(self.bits).y - 1):
                res = self._compute_one_inner_cell(cc, x, y)
                cell_results.append(res)

        return GOL._or(cell_results)

    def _compute_one_inner_cell(self, cc, x, y):
        neighborhood_mask = Num(self.bits).set_neighborhood_of(x, y)

        alive_cell = Num(self.bits).set(x, y)
        dead_cell = Num.zero(self.bits)

        is_cell_alive = GOL._and(cc, alive_cell)
        alive_neighbours = self._count_bits(GOL._and(neighborhood_mask, cc))

        self.__debug_generated_for(x, y)
        return GOL.game_of_live(is_cell_alive, alive_neighbours, alive_cell, dead_cell)

    def _compute_site_cells(self, vars):
        site_cells = [
            self._compute_site_cells_for(Site.TOP,    vars.cc, vars.ct),
            self._compute_site_cells_for(Site.BOTTOM, vars.cc, vars.cb),
            self._compute_site_cells_for(Site.LEFT,   vars.cc, vars.lc),
            self._compute_site_cells_for(Site.RIGHT,  vars.cc, vars.rc),
        ]
        return GOL._or(site_cells)
        
    def _compute_site_cells_for(self, site, cc, site_var):
        cell_results = []

        x_range, y_range = self._get_bounds_for_site(site)

        for x in x_range:
            for y in y_range:
                res = self._compute_one_site_cell(site, cc, site_var, x, y)
                cell_results.append(res)
        
        return GOL._or(cell_results)
    
    def _compute_one_site_cell(self, site, cc, site_var, x, y):
        cc_neighborhood_mask = Num(self.bits).set_neighborhood_of(x, y)
        site_neighborhood_mask = Num(self.bits).set_neighborhood_being(site).neighbor_at(x, y)

        cc_neighborhood = GOL._and(cc_neighborhood_mask, cc)
        site_neighborhood = GOL._and(site_neighborhood_mask, site_var)
        total_neighborhood = GOL._or(cc_neighborhood, site_neighborhood)

        alive_cell = Num(self.bits).set(x, y)
        dead_cell = Num.zero(self.bits)

        is_cell_alive = GOL._and(cc, alive_cell)
        alive_neighbours = self._count_bits(total_neighborhood)

        self.__debug_generated_for(x, y)
        return GOL.game_of_live(is_cell_alive, alive_neighbours, alive_cell, dead_cell)

    def _get_bounds_for_site(self, site):
        y_size = Num(self.bits).y
        x_size = Num(self.bits).x

        if site == Site.TOP:
            xs = 1; xe = x_size - 1
            ys = 0; ye = 1

        elif site == Site.BOTTOM:
            xs = 1; xe = x_size - 1
            ys = y_size - 1; ye = y_size
        
        elif site == Site.LEFT:
            xs = 0; xe = 1
            ys = 1; ye = y_size - 1
        
        elif site == Site.RIGHT:
            xs = x_size - 1; xe = x_size
            ys = 1; ye = y_size - 1

        return range(xs, xe), range(ys, ye)

    def _compute_conner_cells(self, vars):
        conner_cells = [
            self._compute_conner_cells_for([Site.TOP, Site.LEFT],     vars),
            self._compute_conner_cells_for([Site.TOP, Site.RIGHT],    vars),
            self._compute_conner_cells_for([Site.BOTTOM, Site.LEFT],  vars),
            self._compute_conner_cells_for([Site.BOTTOM, Site.RIGHT], vars),
        ]
        return GOL._or(conner_cells)
    
    def _compute_conner_cells_for(self, conner_spec, vars):
        if conner_spec == [Site.TOP, Site.LEFT]:
            x = 0; y = 0
            top_mask = Num(self.bits).set_neighborhood_being(Site.TOP).neighbor_at(x, y)
            left_mask = Num(self.bits).set_neighborhood_being(Site.LEFT).neighbor_at(x, y)
            top_left_mask = Num(self.bits).set_neighborhood_being(Site.TOP, Site.LEFT)

            top_neighborhood = GOL._and(top_mask, vars.ct)
            left_neighborhood = GOL._and(left_mask, vars.lc)
            top_left_neighborhood = GOL._and(top_left_mask, vars.lt)

            total_neighborhood = GOL._or(top_neighborhood, left_neighborhood, top_left_neighborhood)

        elif conner_spec == [Site.TOP, Site.RIGHT]:
            x = Num(self.bits).x - 1; y = 0
            top_mask = Num(self.bits).set_neighborhood_being(Site.TOP).neighbor_at(x, y)
            right_mask = Num(self.bits).set_neighborhood_being(Site.RIGHT).neighbor_at(x, y)
            top_right_mask = Num(self.bits).set_neighborhood_being(Site.TOP, Site.RIGHT)

            top_neighborhood = GOL._and(top_mask, vars.ct)
            right_neighborhood = GOL._and(right_mask, vars.rc)
            top_right_neighborhood = GOL._and(top_right_mask, vars.rt)

            total_neighborhood = GOL._or(top_neighborhood, right_neighborhood, top_right_neighborhood)

        elif conner_spec == [Site.BOTTOM, Site.LEFT]:
            x = 0; y = Num(self.bits).y - 1
            bottom_mask = Num(self.bits).set_neighborhood_being(Site.BOTTOM).neighbor_at(x, y)
            left_mask = Num(self.bits).set_neighborhood_being(Site.LEFT).neighbor_at(x, y)
            bottom_left_mask = Num(self.bits).set_neighborhood_being(Site.BOTTOM, Site.LEFT)

            bottom_neighborhood = GOL._and(bottom_mask, vars.cb)
            left_neighborhood = GOL._and(left_mask, vars.lc)
            bottom_left_neighborhood = GOL._and(bottom_left_mask, vars.lb)

            total_neighborhood = GOL._or(bottom_neighborhood, left_neighborhood, bottom_left_neighborhood)

        elif conner_spec == [Site.BOTTOM, Site.RIGHT]:
            x = Num(self.bits).x - 1; y = Num(self.bits).y - 1
            bottom_mask = Num(self.bits).set_neighborhood_being(Site.BOTTOM).neighbor_at(x, y)
            right_mask = Num(self.bits).set_neighborhood_being(Site.RIGHT).neighbor_at(x, y)
            bottom_right_mask = Num(self.bits).set_neighborhood_being(Site.BOTTOM, Site.RIGHT)

            bottom_neighborhood = GOL._and(bottom_mask, vars.cb)
            right_neighborhood = GOL._and(right_mask, vars.rc)
            bottom_right_neighborhood = GOL._and(bottom_right_mask, vars.rb)

            total_neighborhood = GOL._or(bottom_neighborhood, right_neighborhood, bottom_right_neighborhood)

        cc_neighborhood = GOL._and(Num(self.bits).set_neighborhood_of(x, y), vars.cc)
        total_neighborhood = GOL._or(cc_neighborhood, total_neighborhood)

        alive_cell = Num(self.bits).set(x, y)
        dead_cell = Num.zero(self.bits)

        is_cell_alive = GOL._and(vars.cc, alive_cell)
        alive_neighbours = self._count_bits(total_neighborhood)

        self.__debug_generated_for(x, y)
        return GOL.game_of_live(is_cell_alive, alive_neighbours, alive_cell, dead_cell)
         
    def _count_bits(self, expr):
        return f'POPCOUNT_{self.bits}({expr})'
    
    @staticmethod
    def _or(*args):
        if isinstance(args[0], list):
            args = args[0]
        
        return '(' + '|'.join([f'({a})' for a in args]) + ')'

    @staticmethod
    def _and(*args):
        if isinstance(args[0], list):
            args = args[0]
        
        return '(' + '&'.join([f'({a})' for a in args]) + ')'

    @staticmethod
    def _eq(*args):
        if isinstance(args[0], list):
            args = args[0]
        
        return '(' + '=='.join([f'({a})' for a in args]) + ')'
    
    @staticmethod
    def _shift_left(expr, shift):
        return f'(({expr}) << {shift})'
    
    @staticmethod
    def _shift_right(expr, shift):
        return f'(({expr}) >> {shift})'

    # @staticmethod
    # def game_of_live(cell_is_alive, alive_neighbours, alive_cell, dead_cell):
    #     return GOL.if_else(f'({cell_is_alive})',
    #         GOL.if_else(f'({alive_neighbours} & ~1) == 2', alive_cell, dead_cell),
    #         GOL.if_else(f'({alive_neighbours} == 3)', alive_cell, dead_cell))
    
    # @staticmethod
    # def game_of_live(cell_is_alive, alive_neighbours, alive_cell: Num, dead_cell: Num):

    #     is_alive_expr = GOL._eq(cell_is_alive, alive_cell)
    #     n_is_2 = GOL._eq(alive_neighbours, 2)
    #     n_is_3 = GOL._eq(alive_neighbours, 3)

    #     res = GOL._or(
    #         n_is_3,
    #         GOL._and(is_alive_expr, n_is_2)
    #     )
        
    #     res = Num.cast_expr(res, alive_cell.bit_count)

    #     alive_cell_order = alive_cell.max_order_of_set_bit()

    #     return GOL._shift_left(res, alive_cell_order)

    @staticmethod
    def game_of_live(cell_is_alive, alive_neighbours, alive_cell: Num, dead_cell: Num):

        is_alive_expr = GOL._eq(cell_is_alive, alive_cell)

        n_or_c = GOL._or(
            is_alive_expr,
            alive_neighbours
        )

        res = GOL._eq(n_or_c, 3)
        res = Num.cast_expr(res, alive_cell.bit_count)

        return GOL.if_else(res, alive_cell, dead_cell)

    @staticmethod
    def if_else(condition, true_expr, false_expr):
        return f'(({condition}) ? ({true_expr}) : ({false_expr}))'
    
    def __debug_generated_for(self, x, y):
        self._debug_mask = self._debug_mask.set(x, y , self._debug_mask.bits[y][x] + 1)

def generate_full_macros(vars, *bits_list):
    for bits in bits_list:

        n = Num(bits)
        gol = GOL(bits)

        macro_body = gol.get_macro(VAR)

        args = f'{vars.lt}, {vars.ct}, {vars.rt}, {vars.lc}, {vars.cc}, {vars.rc}, {vars.lb}, {vars.cb}, {vars.rb}'

        print(f'#define __{bits}_BITS__GOL_BITWISE_TILES_COMPUTE({args}) ({macro_body})\n\n')

generate_full_macros(VAR, 16, 32, 64)
