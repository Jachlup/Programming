import numpy as np
import elastica as ea
from pathlib import Path
import csv
import os
from Joint_postprocessing import (
    plot_position,
    plot_video_three_rods,
    plot_video_xy_three_rods,
    plot_video_xz,
)

from FinRay_sim_utils import *


class GripperSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
):
    pass



FOLDER_NAME = "CSV"
CSV_FILE = os.path.join(FOLDER_NAME, "finger_data.csv")


gripper_Sim = GripperSimulator()

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
    n_elem*1,
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
    ea.FixedConstraint, constrained_position_idx=(0,), constrained_director_idx=(0,)
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

output_dir = Path(__file__).resolve().parent
EXPORT_NODE_POSITIONS = True

if EXPORT_NODE_POSITIONS:
    rods_to_export = [rod1, rod2]
    rod_labels = ["rod1", "rod2"]
    export_path = output_dir / "node_positions_after_simulation.csv"
    export_node_positions_csv(export_path, rods_to_export, rod_labels)
    print(f"Saved node positions to: {export_path}")






PLOT_FIGURE = True
SAVE_FIGURE = True
PLOT_VIDEO = True



if PLOT_VIDEO:
    filename = output_dir / "hinge_joint_test.mp4"

    plot_video_xy_three_rods(
        pp_list_rod1,   
        pp_list_rod2,
        pp_list_rod3_all,
        video_name=str(output_dir / "hinge_joint_test_xy.mp4"),
        fps=100,
    )


