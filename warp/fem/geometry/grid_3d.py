import warp as wp

from warp.fem.types import ElementIndex, Coords, Sample, NULL_QP_INDEX, NULL_DOF_INDEX
from warp.fem.types import vec2i, vec3i

from .geometry import Geometry
from .element import Square, Cube


@wp.struct
class Grid3DCellArg:
    res: vec3i
    cell_size: wp.vec3
    origin: wp.vec3


class Grid3D(Geometry):
    """Three-dimensional regular grid geometry"""

    Permutation = wp.types.matrix(shape=(3, 3), dtype=int)
    LOC_TO_WORLD = wp.constant(Permutation(0, 1, 2, 1, 2, 0, 2, 0, 1))
    WORLD_TO_LOC = wp.constant(Permutation(0, 1, 2, 2, 0, 1, 1, 2, 0))

    def __init__(self, res: vec3i, bounds_lo: wp.vec3 = wp.vec3(0.0), bounds_hi: wp.vec3 = wp.vec3(1.0)):
        """Constructs a dense 3D grid

        Args:
            res: Resolution of the grid along each dimension
            bounds_lo: Position of the lower bound of the axis-aligned grid
            bounds_up: Position of the upper bound of the axis-aligned grid
        """

        self.dimension = 3
        self.bounds_lo = bounds_lo
        self.bounds_hi = bounds_hi

        self._res = res

    @property
    def extents(self) -> wp.vec3:
        return self.bounds_hi - self.bounds_lo

    @property
    def cell_size(self) -> wp.vec3:
        ex = self.extents
        return wp.vec3(
            ex[0] / self.res[0],
            ex[1] / self.res[1],
            ex[2] / self.res[2],
        )

    def cell_count(self):
        return self.res[0] * self.res[1] * self.res[2]

    def vertex_count(self):
        return (self.res[0] + 1) * (self.res[1] + 1) * (self.res[2] + 1)

    def side_count(self):
        return (
            (self.res[0] + 1) * (self.res[1]) * (self.res[2])
            + (self.res[0]) * (self.res[1] + 1) * (self.res[2])
            + (self.res[0]) * (self.res[1]) * (self.res[2] + 1)
        )

    def boundary_side_count(self):
        return 2 * (self.res[1]) * (self.res[2]) + (self.res[0]) * 2 * (self.res[2]) + (self.res[0]) * (self.res[1]) * 2

    def reference_cell(self) -> Cube:
        return Cube()

    def reference_side(self) -> Square:
        return Square()

    @property
    def res(self):
        return self._res

    @property
    def origin(self):
        return self.bounds_lo

    @property
    def strides(self):
        return vec3i(self.res[1] * self.res[2], self.res[2], 1)

    # Utility device functions

    CellArg = Grid3DCellArg
    Cell = vec3i

    @wp.func
    def _to_3d_index(strides: vec2i, index: int):
        x = index // strides[0]
        y = (index - strides[0] * x) // strides[1]
        z = index - strides[0] * x - strides[1] * y
        return vec3i(x, y, z)

    @wp.func
    def _from_3d_index(strides: vec2i, index: vec3i):
        return strides[0] * index[0] + strides[1] * index[1] + index[2]

    @wp.func
    def cell_index(res: vec3i, cell: Cell):
        strides = vec2i(res[1] * res[2], res[2])
        return Grid3D._from_3d_index(strides, cell)

    @wp.func
    def get_cell(res: vec3i, cell_index: ElementIndex):
        strides = vec2i(res[1] * res[2], res[2])
        return Grid3D._to_3d_index(strides, cell_index)

    @wp.struct
    class Side:
        axis: int  # normal
        origin: vec3i  # index of vertex at corner (0,0,0)

    @wp.struct
    class SideArg:
        cell_count: int
        axis_offsets: vec3i
        cell_arg: Grid3DCellArg

    SideIndexArg = SideArg

    @wp.func
    def _world_to_local(axis: int, vec: vec3i):
        return vec3i(
            vec[Grid3D.LOC_TO_WORLD[axis, 0]],
            vec[Grid3D.LOC_TO_WORLD[axis, 1]],
            vec[Grid3D.LOC_TO_WORLD[axis, 2]],
        )

    @wp.func
    def _local_to_world(axis: int, vec: vec3i):
        return vec3i(
            vec[Grid3D.WORLD_TO_LOC[axis, 0]],
            vec[Grid3D.WORLD_TO_LOC[axis, 1]],
            vec[Grid3D.WORLD_TO_LOC[axis, 2]],
        )

    @wp.func
    def _local_to_world(axis: int, vec: wp.vec3):
        return wp.vec3(
            vec[Grid3D.WORLD_TO_LOC[axis, 0]],
            vec[Grid3D.WORLD_TO_LOC[axis, 1]],
            vec[Grid3D.WORLD_TO_LOC[axis, 2]],
        )

    @wp.func
    def side_index(arg: SideArg, side: Side):
        alt_axis = Grid3D.LOC_TO_WORLD[side.axis, 0]
        if side.origin[0] == arg.cell_arg.res[alt_axis]:
            # Upper-boundary side
            longitude = side.origin[1]
            latitude = side.origin[2]

            latitude_res = arg.cell_arg.res[Grid3D.LOC_TO_WORLD[side.axis, 2]]
            lat_long = latitude_res * longitude + latitude

            return 3 * arg.cell_count + arg.axis_offsets[side.axis] + lat_long

        cell_index = Grid3D.cell_index(arg.cell_arg.res, Grid3D._local_to_world(side.axis, side.origin))
        return side.axis * arg.cell_count + cell_index

    @wp.func
    def get_side(arg: SideArg, side_index: ElementIndex):
        if side_index < 3 * arg.cell_count:
            axis = side_index // arg.cell_count
            cell_index = side_index - axis * arg.cell_count
            origin = Grid3D._world_to_local(axis, Grid3D.get_cell(arg.cell_arg.res, cell_index))
            return Grid3D.Side(axis, origin)

        axis_side_index = side_index - 3 * arg.cell_count
        if axis_side_index < arg.axis_offsets[1]:
            axis = 0
        elif axis_side_index < arg.axis_offsets[2]:
            axis = 1
        else:
            axis = 2

        altitude = arg.cell_arg.res[Grid3D.LOC_TO_WORLD[axis, 0]]

        lat_long = axis_side_index - arg.axis_offsets[axis]
        latitude_res = arg.cell_arg.res[Grid3D.LOC_TO_WORLD[axis, 2]]

        longitude = lat_long // latitude_res
        latitude = lat_long - longitude * latitude_res

        origin_loc = vec3i(altitude, longitude, latitude)

        return Grid3D.Side(axis, origin_loc)

    # Geometry device interface

    def cell_arg_value(self, device) -> CellArg:
        args = self.CellArg()
        args.res = self.res
        args.cell_size = self.cell_size
        args.origin = self.bounds_lo
        return args

    @wp.func
    def cell_position(args: CellArg, s: Sample):
        cell = Grid3D.get_cell(args.res, s.element_index)
        return (
            wp.vec3(
                (float(cell[0]) + s.element_coords[0]) * args.cell_size[0],
                (float(cell[1]) + s.element_coords[1]) * args.cell_size[1],
                (float(cell[2]) + s.element_coords[2]) * args.cell_size[2],
            )
            + args.origin
        )

    @wp.func
    def cell_lookup(args: CellArg, pos: wp.vec3):
        loc_pos = wp.cw_div(pos - args.origin, args.cell_size)
        x = wp.clamp(loc_pos[0], 0.0, float(args.res[0]))
        y = wp.clamp(loc_pos[1], 0.0, float(args.res[1]))
        z = wp.clamp(loc_pos[2], 0.0, float(args.res[2]))

        x_cell = wp.min(wp.floor(x), float(args.res[0]) - 1.0)
        y_cell = wp.min(wp.floor(y), float(args.res[1]) - 1.0)
        z_cell = wp.min(wp.floor(z), float(args.res[2]) - 1.0)

        coords = Coords(x - x_cell, y - y_cell, z - z_cell)
        cell_index = Grid3D.cell_index(args.res, Grid3D.Cell(int(x_cell), int(y_cell), int(z_cell)))

        return Sample(cell_index, coords, NULL_QP_INDEX, 0.0, NULL_DOF_INDEX, NULL_DOF_INDEX)

    @wp.func
    def cell_lookup(args: CellArg, pos: wp.vec3, guess: Sample):
        return Grid3D.cell_lookup(args, pos)

    @wp.func
    def cell_measure(args: CellArg, cell_index: ElementIndex, coords: Coords):
        return args.cell_size[0] * args.cell_size[1] * args.cell_size[2]

    @wp.func
    def cell_measure(args: CellArg, s: Sample):
        return Grid3D.cell_measure(args, s.element_index, s.element_coords)

    @wp.func
    def cell_measure_ratio(args: CellArg, s: Sample):
        return 1.0

    @wp.func
    def cell_normal(args: CellArg, s: Sample):
        return wp.vec3(0.0)

    def side_arg_value(self, device) -> SideArg:
        args = self.SideArg()

        axis_dims = vec3i(
            self.res[1] * self.res[2],
            self.res[2] * self.res[0],
            self.res[0] * self.res[1],
        )
        args.axis_offsets = vec3i(
            0,
            axis_dims[0],
            axis_dims[0] + axis_dims[1],
        )
        args.cell_count = self.cell_count()
        args.cell_arg = self.cell_arg_value(device)
        return args

    def side_index_arg_value(self, device) -> SideIndexArg:
        return self.side_arg_value(device)

    @wp.func
    def boundary_side_index(args: SideArg, boundary_side_index: int):
        """Boundary side to side index"""

        axis_side_index = boundary_side_index // 2
        border = boundary_side_index - 2 * axis_side_index

        if axis_side_index < args.axis_offsets[1]:
            axis = 0
        elif axis_side_index < args.axis_offsets[2]:
            axis = 1
        else:
            axis = 2

        lat_long = axis_side_index - args.axis_offsets[axis]
        latitude_res = args.cell_arg.res[Grid3D.LOC_TO_WORLD[axis, 2]]

        longitude = lat_long // latitude_res
        latitude = lat_long - longitude * latitude_res

        altitude = border * args.cell_arg.res[axis]

        side = Grid3D.Side(axis, vec3i(altitude, longitude, latitude))
        return Grid3D.side_index(args, side)

    @wp.func
    def side_position(args: SideArg, s: Sample):
        side = Grid3D.get_side(args, s.element_index)

        local_pos = wp.vec3(
            float(side.origin[0]),
            float(side.origin[1]) + s.element_coords[0],
            float(side.origin[2]) + s.element_coords[1],
        )

        pos = args.cell_arg.origin + wp.cw_mul(Grid3D._local_to_world(side.axis, local_pos), args.cell_arg.cell_size)

        return pos

    @wp.func
    def side_measure(args: SideArg, side_index: ElementIndex, coords: Coords):
        side = Grid3D.get_side(args, side_index)
        long_axis = Grid3D.LOC_TO_WORLD[side.axis, 1]
        lat_axis = Grid3D.LOC_TO_WORLD[side.axis, 2]
        return args.cell_arg.cell_size[long_axis] * args.cell_arg.cell_size[lat_axis]

    @wp.func
    def side_measure(args: SideArg, s: Sample):
        return Grid3D.side_measure(args, s.element_index, s.element_coords)

    @wp.func
    def side_measure_ratio(args: SideArg, s: Sample):
        side = Grid3D.get_side(args, s.element_index)
        alt_axis = Grid3D.LOC_TO_WORLD[side.axis, 0]
        return 1.0 / args.cell_arg.cell_size[alt_axis]

    @wp.func
    def side_normal(args: SideArg, s: Sample):
        side = Grid3D.get_side(args, s.element_index)

        if side.origin[0] == 0:
            sign = -1.0
        else:
            sign = 1.0

        local_n = wp.vec3(sign, 0.0, 0.0)
        return Grid3D._local_to_world(side.axis, local_n)

    @wp.func
    def side_inner_cell_index(arg: SideArg, side_index: ElementIndex):
        side = Grid3D.get_side(arg, side_index)

        if side.origin[0] == 0:
            inner_alt = 0
        else:
            inner_alt = side.origin[0] - 1

        inner_origin = vec3i(inner_alt, side.origin[1], side.origin[2])

        cell = Grid3D._local_to_world(side.axis, inner_origin)
        return Grid3D.cell_index(arg.cell_arg.res, cell)

    @wp.func
    def side_outer_cell_index(arg: SideArg, side_index: ElementIndex):
        side = Grid3D.get_side(arg, side_index)

        alt_axis = Grid3D.LOC_TO_WORLD[side.axis, 0]

        if side.origin[0] == arg.cell_arg.res[alt_axis]:
            outer_alt = arg.cell_arg.res[alt_axis] - 1
        else:
            outer_alt = side.origin[0]

        outer_origin = vec3i(outer_alt, side.origin[1], side.origin[2])

        cell = Grid3D._local_to_world(side.axis, outer_origin)
        return Grid3D.cell_index(arg.cell_arg.res, cell)
