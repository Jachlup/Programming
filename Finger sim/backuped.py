

__doc__ = """Hinge joint example, for detailed explanation refer to Zhang et. al. Nature Comm.  methods section."""

import numpy as np
import elastica as ea
from pathlib import Path
from Joint_postprocessing import (
    plot_position,
    plot_video_three_rods,
    plot_video_xy_three_rods,
    plot_video_xz,
)


class HingeJointSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
):
    pass


class NodeForce(ea.NoForces):
    def __init__(self, force, node_idx: int):
        super().__init__()
        self.force = np.asarray(force, dtype=float)
        self.node_idx = int(node_idx)

    def apply_forces(self, system, time=np.float64(0.0)) -> None:
        system.external_forces[..., self.node_idx] += self.force


def endpoints_to_vector_and_length(node_a, node_b):
    direction_vector = node_b - node_a
    length = np.linalg.norm(direction_vector)
    if length == 0.0:
        raise ValueError("Rod endpoints must be different points.")
    direction = direction_vector / length
    return node_a, direction, length


hinge_joint_sim = HingeJointSimulator()

# setting up test params
n_elem = 8
normal = np.array([0.0, 0.0, 1.0])
roll_direction = normal.copy()
base_radius = 0.007
base_area = np.pi * base_radius**2
density = 1750
E = 2e6
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)
E_rod3 = 2e5
shear_modulus_rod3 = E_rod3 / (poisson_ratio + 1.0)
num_parallel_rod3 = 10
beam_attach_span_nodes = 5

start_rod_1 = np.array([0.0, 0.0, 0.0])
end_rod_1 = np.array([0.0, 0.1, 0.0])
start_rod_1, direction_rod1, base_length = endpoints_to_vector_and_length(
    start_rod_1, end_rod_1
)

attach_idx_rod1 = n_elem // 2
attach_pos_rod1 = (
    start_rod_1 + direction_rod1 * (base_length * attach_idx_rod1 / n_elem)
)

start_rod_2 = np.array([0.03, 0.0045, 0.0])
end_rod_2 = end_rod_1.copy()
start_rod_2, direction_rod2, base_length_rod2 = endpoints_to_vector_and_length(
    start_rod_2, end_rod_2
)

attach_idx_rod2 = n_elem // 2
attach_pos_rod2 = (
    start_rod_2 + direction_rod2 * (base_length_rod2 * attach_idx_rod2 / n_elem)
)

attach_offsets = np.rint(
    np.linspace(-beam_attach_span_nodes / 2, beam_attach_span_nodes / 2, num_parallel_rod3)
).astype(int)
attach_idx_rod1_all = np.clip(attach_idx_rod1 + attach_offsets, 0, n_elem)
attach_idx_rod2_all = np.clip(attach_idx_rod2 + attach_offsets, 0, n_elem)

# Create rod 1
rod1 = ea.CosseratRod.straight_rod(
    n_elem,
    start_rod_1,
    direction_rod1,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
hinge_joint_sim.append(rod1)
# Create rod 2
normal = np.array([0.0, 0.0, 1.0])
rod2 = ea.CosseratRod.straight_rod(
    n_elem,
    start_rod_2,
    direction_rod2,
    normal,
    base_length_rod2,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
hinge_joint_sim.append(rod2)

# Create multiple rod3-like rods between distributed rod1/rod2 attachment points
rod3_list = []
for attach_idx_rod2_i, attach_idx_rod1_i in zip(attach_idx_rod2_all, attach_idx_rod1_all):
    attach_pos_rod2_i = start_rod_2 + direction_rod2 * (
        base_length_rod2 * int(attach_idx_rod2_i) / n_elem
    )
    attach_pos_rod1_i = start_rod_1 + direction_rod1 * (
        base_length * int(attach_idx_rod1_i) / n_elem
    )

    start_rod_3_i = attach_pos_rod2_i.copy()
    end_rod_3_i = attach_pos_rod1_i.copy()
    start_rod_3_i, direction_rod3_i, base_length_rod3_i = endpoints_to_vector_and_length(
        start_rod_3_i, end_rod_3_i
    )

    rod3_i = ea.CosseratRod.straight_rod(
        n_elem,
        start_rod_3_i,
        direction_rod3_i,
        normal,
        base_length_rod3_i,
        base_radius,
        density,
        youngs_modulus=E_rod3,
        shear_modulus=shear_modulus_rod3,
    )
    hinge_joint_sim.append(rod3_i)
    rod3_list.append(rod3_i)

# Apply boundary conditions to rod1.
hinge_joint_sim.constrain(rod1).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)
hinge_joint_sim.constrain(rod2).using(
    ea.GeneralConstraint,
    rod2.position_collection[..., 0].copy(),
    constrained_position_idx=(0,),
    translational_constraint_selector=np.array([True, True, True]),
)

# Connect rod 1 and rod 2
hinge_joint_sim.connect(
    first_rod=rod1, second_rod=rod2, first_connect_idx=-1, second_connect_idx=-1
).using(
    ea.HingeJoint, k=5e4, nu=50.0, kt=0.0, normal_direction=roll_direction
)  # 1e-2

# Connect rod 2/rod 1 to every rod3-like rod using distributed attachment nodes
for rod3_i, attach_idx_rod2_i, attach_idx_rod1_i in zip(
    rod3_list, attach_idx_rod2_all, attach_idx_rod1_all
):
    hinge_joint_sim.connect(
        first_rod=rod2,
        second_rod=rod3_i,
        first_connect_idx=int(attach_idx_rod2_i),
        second_connect_idx=0,
    ).using(ea.HingeJoint, k=5e3, nu=20.0, kt=0.0, normal_direction=roll_direction)

    hinge_joint_sim.connect(
        first_rod=rod1,
        second_rod=rod3_i,
        first_connect_idx=int(attach_idx_rod1_i),
        second_connect_idx=-1,
    ).using(ea.HingeJoint, k=5e3, nu=20.0, kt=0.0, normal_direction=roll_direction)

# Add forces to rod2
hinge_joint_sim.add_forcing_to(rod2).using(
    NodeForce,
    force=np.array([0.5, 0.0, 0.0]),
    node_idx=1,
)

# add damping
damping_constant = 5e-2
dt = 1e-5
hinge_joint_sim.dampen(rod1).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
hinge_joint_sim.dampen(rod2).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
for rod3_i in rod3_list:
    hinge_joint_sim.dampen(rod3_i).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )

pp_list_rod1 = ea.defaultdict(list)
pp_list_rod2 = ea.defaultdict(list)
pp_list_rod3_all = [ea.defaultdict(list) for _ in range(num_parallel_rod3)]

hinge_joint_sim.collect_diagnostics(rod1).using(
    ea.MyCallBack, step_skip=1000, callback_params=pp_list_rod1
)
hinge_joint_sim.collect_diagnostics(rod2).using(
    ea.MyCallBack, step_skip=1000, callback_params=pp_list_rod2
)
for rod3_i, pp_list_rod3_i in zip(rod3_list, pp_list_rod3_all):
    hinge_joint_sim.collect_diagnostics(rod3_i).using(
        ea.MyCallBack, step_skip=1000, callback_params=pp_list_rod3_i
    )

hinge_joint_sim.finalize()
timestepper = ea.PositionVerlet()
# timestepper = PEFRL()

final_time = 0.2
dl = base_length / n_elem
total_steps = int(final_time / dt)
print("Total steps", total_steps)
ea.integrate(timestepper, hinge_joint_sim, final_time, total_steps)

PLOT_FIGURE = True
SAVE_FIGURE = True
PLOT_VIDEO = True

# plotting results
# if PLOT_FIGURE:
#     filename = "hinge_joint_test.png"
#     plot_position(pp_list_rod1, pp_list_rod2, filename, SAVE_FIGURE)

if PLOT_VIDEO:
    output_dir = Path(__file__).resolve().parent
    filename = output_dir / "hinge_joint_test.mp4"
    # plot_video_three_rods(
    #     pp_list_rod1,
    #     pp_list_rod2,
    #     pp_list_rod3,
    #     video_name=str(filename),
    #     fps=100,
    # )
    plot_video_xy_three_rods(
        pp_list_rod1,
        pp_list_rod2,
        pp_list_rod3_all,
        video_name=str(output_dir / "hinge_joint_test_xy.mp4"),
        fps=100,
    )
    # plot_video_xz(
    #     pp_list_rod1, pp_list_rod2, video_name=filename + "_xz.mp4", margin=0.2, fps=100
    # )


"""sth works 14:36 19 Mar"""

import numpy as np
import elastica as ea
from pathlib import Path
from Joint_postprocessing import (
    plot_position,
    plot_video_three_rods,
    plot_video_xy_three_rods,
    plot_video_xz,
)


class HingeJointSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
):
    pass


class NodeForce(ea.NoForces):
    def __init__(self, force, node_idx: int):
        super().__init__()
        self.force = np.asarray(force, dtype=float)
        self.node_idx = int(node_idx)

    def apply_forces(self, system, time=np.float64(0.0)) -> None:
        system.external_forces[..., self.node_idx] += self.force


def endpoints_to_vector_and_length(node_a, node_b):
    direction_vector = node_b - node_a
    length = np.linalg.norm(direction_vector)
    if length == 0.0:
        raise ValueError("Rod endpoints must be different points.")
    direction = direction_vector / length
    return node_a, direction, length


hinge_joint_sim = HingeJointSimulator()

# setting up test params
n_elem = 10
n_elem_rod3 = 5
normal = np.array([0.0, 0.0, 1.0])
roll_direction = normal.copy()
base_radius = 0.007
base_area = np.pi * base_radius**2
density = 1750
E = 2e6
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)
E_rod3 = 2e5
shear_modulus_rod3 = E_rod3 / (poisson_ratio + 1.0)
num_parallel_rod3 = 10
beam_attach_span_nodes = 9

start_rod_1 = np.array([0.0, 0.0, 0.0])
end_rod_1 = np.array([0.0, 0.1, 0.0])
start_rod_1, direction_rod1, base_length = endpoints_to_vector_and_length(
    start_rod_1, end_rod_1
)

attach_idx_rod1 = n_elem // 2
attach_pos_rod1 = (
    start_rod_1 + direction_rod1 * (base_length * attach_idx_rod1 / n_elem)
)

start_rod_2 = np.array([0.03, 0.0045, 0.0])
end_rod_2 = end_rod_1.copy()
start_rod_2, direction_rod2, base_length_rod2 = endpoints_to_vector_and_length(
    start_rod_2, end_rod_2
)

attach_idx_rod2 = n_elem // 2
attach_pos_rod2 = (
    start_rod_2 + direction_rod2 * (base_length_rod2 * attach_idx_rod2 / n_elem)
)

# Evenly distribute the attachment indices over the rod's length to prevent overlaps.
# We go up to n_elem - 1 to avoid the very end node where rod1 and rod2 touch.
attach_idx_rod1_all = np.linspace(0, n_elem - 1, num_parallel_rod3).astype(int)
attach_idx_rod2_all = np.linspace(0, n_elem - 1, num_parallel_rod3).astype(int)

# Create rod 1
rod1 = ea.CosseratRod.straight_rod(
    n_elem,
    start_rod_1,
    direction_rod1,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
hinge_joint_sim.append(rod1)
# Create rod 2
normal = np.array([0.0, 0.0, 1.0])
rod2 = ea.CosseratRod.straight_rod(
    n_elem,
    start_rod_2,
    direction_rod2,
    normal,
    base_length_rod2,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
hinge_joint_sim.append(rod2)
# Create multiple rod3-like rods between distributed rod1/rod2 attachment points
rod3_list = []
for idx2, idx1 in zip(attach_idx_rod2_all, attach_idx_rod1_all):
    start_pos = start_rod_2 + direction_rod2 * (base_length_rod2 * idx2 / n_elem)
    end_pos = start_rod_1 + direction_rod1 * (base_length * idx1 / n_elem)
    
    start_rod3, direction_rod3, length_rod3 = endpoints_to_vector_and_length(start_pos, end_pos)
    
    rod3 = ea.CosseratRod.straight_rod(
        n_elem_rod3, start_rod3, direction_rod3, normal, length_rod3,
        base_radius, density, youngs_modulus=E_rod3, shear_modulus=shear_modulus_rod3,
    )
    hinge_joint_sim.append(rod3)
    rod3_list.append(rod3)

# Apply boundary conditions to rod1.
hinge_joint_sim.constrain(rod1).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)
hinge_joint_sim.constrain(rod2).using(
    ea.GeneralConstraint,
    rod2.position_collection[..., 0].copy(),
    constrained_position_idx=(0,),
    translational_constraint_selector=np.array([True, True, True]),
)

# Connect rod 1 and rod 2
hinge_joint_sim.connect(
    first_rod=rod1, second_rod=rod2, first_connect_idx=-1, second_connect_idx=-1
).using(
    ea.HingeJoint, k=1e4, nu=1.0, kt=0.0, normal_direction=roll_direction
)  # 1e-2

# Connect rod 2/rod 1 to every rod3-like rod using distributed attachment nodes
for rod3_i, attach_idx_rod2_i, attach_idx_rod1_i in zip(
    rod3_list, attach_idx_rod2_all, attach_idx_rod1_all
):
    hinge_joint_sim.connect(
        first_rod=rod2,
        second_rod=rod3_i,
        first_connect_idx=int(attach_idx_rod2_i),
        second_connect_idx=0,
    ).using(ea.HingeJoint, k=1e4, nu=1.0, kt=0.0, normal_direction=roll_direction)

    hinge_joint_sim.connect(
        first_rod=rod1,
        second_rod=rod3_i,
        first_connect_idx=int(attach_idx_rod1_i),
        second_connect_idx=-1,
    ).using(ea.HingeJoint, k=1e4, nu=1.0, kt=0.0, normal_direction=roll_direction)

# Add forces to rod2
hinge_joint_sim.add_forcing_to(rod2).using(
    NodeForce,
    force=np.array([0.5, 0.0, 0.0]),
    node_idx=1,
)

# add damping
damping_constant = 5e-2
dt = 5e-5
hinge_joint_sim.dampen(rod1).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
hinge_joint_sim.dampen(rod2).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
for rod3_i in rod3_list:
    hinge_joint_sim.dampen(rod3_i).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )

pp_list_rod1 = ea.defaultdict(list)
pp_list_rod2 = ea.defaultdict(list)
pp_list_rod3_all = [ea.defaultdict(list) for _ in range(num_parallel_rod3)]

hinge_joint_sim.collect_diagnostics(rod1).using(
    ea.MyCallBack, step_skip=200, callback_params=pp_list_rod1 #step_skip - how often sim saves data
)
hinge_joint_sim.collect_diagnostics(rod2).using(
    ea.MyCallBack, step_skip=200, callback_params=pp_list_rod2
)
for rod3_i, pp_list_rod3_i in zip(rod3_list, pp_list_rod3_all):
    hinge_joint_sim.collect_diagnostics(rod3_i).using(
        ea.MyCallBack, step_skip=200, callback_params=pp_list_rod3_i
    )

hinge_joint_sim.finalize()
timestepper = ea.PositionVerlet()
# timestepper = PEFRL()

final_time = 2
dl = base_length / n_elem
total_steps = int(final_time / dt)
print("Total steps", total_steps)
ea.integrate(timestepper, hinge_joint_sim, final_time, total_steps)

PLOT_FIGURE = True
SAVE_FIGURE = True
PLOT_VIDEO = True

# plotting results
# if PLOT_FIGURE:
#     filename = "hinge_joint_test.png"
#     plot_position(pp_list_rod1, pp_list_rod2, filename, SAVE_FIGURE)

if PLOT_VIDEO:
    output_dir = Path(__file__).resolve().parent
    filename = output_dir / "hinge_joint_test.mp4"
    # plot_video_three_rods(
    #     pp_list_rod1,
    #     pp_list_rod2,
    #     pp_list_rod3,
    #     video_name=str(filename),
    #     fps=100,
    # )
    plot_video_xy_three_rods(
        pp_list_rod1,
        pp_list_rod2,
        pp_list_rod3_all,
        video_name=str(output_dir / "hinge_joint_test_xy.mp4"),
        fps=100,
    )
    # plot_video_xz(
    #     pp_list_rod1, pp_list_rod2, video_name=filename + "_xz.mp4", margin=0.2, fps=100
    # )

"""asdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfsdssssssssssssssssssssssssssssssssssssssssssssssssssssss"""

import numpy as np
import elastica as ea
from pathlib import Path
from Joint_postprocessing import (
    plot_position,
    plot_video_three_rods,
    plot_video_xy_three_rods,
    plot_video_xz,
)


class HingeJointSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
):
    pass


class NodeForce(ea.NoForces):
    def __init__(self, force, node_idx: int):
        super().__init__()
        self.force = np.asarray(force, dtype=float)
        self.node_idx = int(node_idx)

    def apply_forces(self, system, time=np.float64(0.0)) -> None:
        system.external_forces[..., self.node_idx] += self.force


def endpoints_to_vector_and_length(node_a, node_b):
    direction_vector = node_b - node_a
    length = np.linalg.norm(direction_vector)
    if length == 0.0:
        raise ValueError("Rod endpoints must be different points.")
    direction = direction_vector / length
    return node_a, direction, length


gripper_Sim = HingeJointSimulator()

# setting up test params
n_elem = 10
n_elem_rod3 = 5
normal = np.array([0.0, 0.0, 1.0])
roll_direction = normal.copy()
base_radius = 0.007
base_area = np.pi * base_radius**2
density = 1750
E = 2e6
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)
E_rod3 = 2e5
shear_modulus_rod3 = E_rod3 / (poisson_ratio + 1.0)
num_parallel_rod3 = 10
beam_attach_span_nodes = 9

start_rod_1 = np.array([0.0, 0.0, 0.0])
end_rod_1 = np.array([0.0, 0.1, 0.0])
start_rod_1, direction_rod1, base_length = endpoints_to_vector_and_length(
    start_rod_1, end_rod_1
)

attach_idx_rod1 = n_elem // 2
attach_pos_rod1 = (
    start_rod_1 + direction_rod1 * (base_length * attach_idx_rod1 / n_elem)
)

start_rod_2 = np.array([0.03, 0.0045, 0.0])
end_rod_2 = end_rod_1.copy()
start_rod_2, direction_rod2, base_length_rod2 = endpoints_to_vector_and_length(
    start_rod_2, end_rod_2
)

attach_idx_rod2 = n_elem // 2
attach_pos_rod2 = (
    start_rod_2 + direction_rod2 * (base_length_rod2 * attach_idx_rod2 / n_elem)
)

# Evenly distribute the attachment indices over the rod's length to prevent overlaps.
# We go up to n_elem - 1 to avoid the very end node where rod1 and rod2 touch.
attach_idx_rod1_all = np.linspace(0, n_elem - 1, num_parallel_rod3).astype(int)
attach_idx_rod2_all = np.linspace(0, n_elem - 1, num_parallel_rod3).astype(int)

# Create rod 1
rod1 = ea.CosseratRod.straight_rod(
    n_elem,
    start_rod_1,
    direction_rod1,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
gripper_Sim.append(rod1)
# Create rod 2
normal = np.array([0.0, 0.0, 1.0])
rod2 = ea.CosseratRod.straight_rod(
    n_elem,
    start_rod_2,
    direction_rod2,
    normal,
    base_length_rod2,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
gripper_Sim.append(rod2)
# Create multiple rod3-like rods between distributed rod1/rod2 attachment points
rod3_list = []
for idx2, idx1 in zip(attach_idx_rod2_all, attach_idx_rod1_all):
    start_pos = start_rod_2 + direction_rod2 * (base_length_rod2 * idx2 / n_elem)
    end_pos = start_rod_1 + direction_rod1 * (base_length * idx1 / n_elem)
    
    start_rod3, direction_rod3, length_rod3 = endpoints_to_vector_and_length(start_pos, end_pos)
    
    rod3 = ea.CosseratRod.straight_rod(
        n_elem_rod3, start_rod3, direction_rod3, normal, length_rod3,
        base_radius, density, youngs_modulus=E_rod3, shear_modulus=shear_modulus_rod3,
    )
    gripper_Sim.append(rod3)
    rod3_list.append(rod3)

# Apply boundary conditions to rod1.
gripper_Sim.constrain(rod1).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)
gripper_Sim.constrain(rod2).using(
    ea.GeneralConstraint,
    rod2.position_collection[..., 0].copy(),
    constrained_position_idx=(0,),
    translational_constraint_selector=np.array([True, True, True]),
)

# Connections

# Connect rod 1 and rod 2
gripper_Sim.connect(
    first_rod=rod1, second_rod=rod2, first_connect_idx=-1, second_connect_idx=-1
).using(
    ea.HingeJoint, 
    k=1e4, 
    nu=1.0, 
    kt=0.0, 
    normal_direction=roll_direction
)  

# Connect rod 2/rod 1 to every rod3-like rod using distributed attachment nodes
for rod3_i, attach_idx_rod2_i, attach_idx_rod1_i in zip(
    rod3_list, attach_idx_rod2_all, attach_idx_rod1_all
):
    gripper_Sim.connect(
        first_rod=rod2,
        second_rod=rod3_i,
        first_connect_idx=int(attach_idx_rod2_i),
        second_connect_idx=0,
    ).using(ea.HingeJoint, k=1e4, nu=1.0, kt=0.0, normal_direction=roll_direction)

    gripper_Sim.connect(
        first_rod=rod1,
        second_rod=rod3_i,
        first_connect_idx=int(attach_idx_rod1_i),
        second_connect_idx=-1,
    ).using(ea.HingeJoint, k=1e4, nu=1.0, kt=0.0, normal_direction=roll_direction)


force_point = rod3_list[5]

# Add forces to rod2
gripper_Sim.add_forcing_to(force_point).using(
    NodeForce,
    force=np.array([10.0, 0.0, 0.0]), # N
    node_idx=1,
)

# add damping
damping_constant = 5e-2
dt = 5e-5
gripper_Sim.dampen(rod1).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
gripper_Sim.dampen(rod2).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
for rod3_i in rod3_list:
    gripper_Sim.dampen(rod3_i).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=dt,
    )



pp_list_rod1 = ea.defaultdict(list)
pp_list_rod2 = ea.defaultdict(list)
pp_list_rod3_all = [ea.defaultdict(list) for _ in range(num_parallel_rod3)]

gripper_Sim.collect_diagnostics(rod1).using(
    ea.MyCallBack, step_skip=200, callback_params=pp_list_rod1 #step_skip - how often sim saves data
)
gripper_Sim.collect_diagnostics(rod2).using(
    ea.MyCallBack, step_skip=200, callback_params=pp_list_rod2
)
for rod3_i, pp_list_rod3_i in zip(rod3_list, pp_list_rod3_all):
    gripper_Sim.collect_diagnostics(rod3_i).using(
        ea.MyCallBack, step_skip=200, callback_params=pp_list_rod3_i
    )

gripper_Sim.finalize()
timestepper = ea.PositionVerlet()


final_time = 2
dl = base_length / n_elem
total_steps = int(final_time / dt)
print("Total steps", total_steps)
ea.integrate(timestepper, gripper_Sim, final_time, total_steps)

PLOT_FIGURE = True
SAVE_FIGURE = True
PLOT_VIDEO = True



if PLOT_VIDEO:
    output_dir = Path(__file__).resolve().parent
    filename = output_dir / "hinge_joint_test.mp4"

    plot_video_xy_three_rods(
        pp_list_rod1,
        pp_list_rod2,
        pp_list_rod3_all,
        video_name=str(output_dir / "hinge_joint_test_xy.mp4"),
        fps=100,
    )


