import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from scipy.spatial.transform import Rotation


DOT_SIZE = 2
BEAM_WIDTH = 3
ZOOM_XY_XLIM = (-0.05, 0.08)
ZOOM_XY_YLIM = (-0.01, 0.11)
ZOOM_3D_ZLIM = (-0.02, 0.02)


def _make_animation_writer(manimation, fps):
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    available_writers = set(manimation.writers.list())
    if "ffmpeg" in available_writers:
        return manimation.writers["ffmpeg"](fps=fps, metadata=metadata), "ffmpeg"
    if "pillow" in available_writers:
        return manimation.PillowWriter(fps=fps), "pillow"
    raise RuntimeError(
        "No supported MovieWriter found. Install ffmpeg or pillow for video export."
    )


def _resolve_video_name_for_writer(video_name, writer_name):
    if writer_name == "pillow" and str(video_name).lower().endswith(".mp4"):
        return str(video_name)[:-4] + ".gif"
    return str(video_name)


def _position_collections_from_input(plot_params_rod3):
    if isinstance(plot_params_rod3, dict):
        return [np.array(plot_params_rod3["position"])]
    return [np.array(plot_params["position"]) for plot_params in plot_params_rod3]


def plot_position(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    filename="spherical_joint_test.png",
    SAVE_FIGURE=False,
):

    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])

    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)
    ax = fig.add_subplot(111)

    ax.grid(which="minor", color="k", linestyle="--")
    ax.grid(which="major", color="k", linestyle="-")
    ax.plot(
        position_of_rod1[:, 0, -1],
        position_of_rod1[:, 1, -1],
        "r-",
        linewidth=BEAM_WIDTH,
        label="rod1",
    )
    ax.plot(
        position_of_rod2[:, 0, -1],
        position_of_rod2[:, 1, -1],
        c=to_rgb("xkcd:bluish"),
        linewidth=BEAM_WIDTH,
        label="rod2",
    )

    fig.legend(prop={"size": 20})

    plt.show()

    if SAVE_FIGURE:
        fig.savefig(filename)


def plot_orientation(title, time, directors):
    quat = []
    for t in range(len(time)):
        quat_t = Rotation.from_matrix(directors[t].T).as_quat()
        quat.append(quat_t)
    quat = np.array(quat)

    plt.figure(num=title)
    plt.plot(time, quat[:, 0], label="x")
    plt.plot(time, quat[:, 1], label="y")
    plt.plot(time, quat[:, 2], label="z")
    plt.plot(time, quat[:, 3], label="w")
    plt.title(title)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Quaternion")
    plt.show()


def plot_video(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])

    writer, writer_name = _make_animation_writer(manimation, fps)
    print(f"plot video (writer: {writer_name})")
    target_video_name = _resolve_video_name_for_writer(video_name, writer_name)
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    with writer.saving(fig, target_video_name, 100):
        for time in range(1, len(time)):
            fig.clf()
            ax = plt.axes(projection="3d")  # fig.add_subplot(111)
            ax.grid(which="minor", color="k", linestyle="--")
            ax.grid(which="major", color="k", linestyle="-")
            ax.plot(
                position_of_rod1[time, 0],
                position_of_rod1[time, 1],
                position_of_rod1[time, 2],
                color="r",
                marker="o",
                markersize=DOT_SIZE,
                linewidth=BEAM_WIDTH,
                label="rod1",
            )
            ax.plot(
                position_of_rod2[time, 0],
                position_of_rod2[time, 1],
                position_of_rod2[time, 2],
                color=to_rgb("xkcd:bluish"),
                marker="o",
                markersize=DOT_SIZE,
                linewidth=BEAM_WIDTH,
                label="rod2",
            )

            ax.set_xlim(-0.25, 0.25)
            ax.set_ylim(-0.25, 0.25)
            ax.set_zlim(0, 0.4)
            writer.grab_frame()


def plot_video_xy(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])

    writer, writer_name = _make_animation_writer(manimation, fps)
    print(f"plot video xy (writer: {writer_name})")
    target_video_name = _resolve_video_name_for_writer(video_name, writer_name)
    fig = plt.figure()
    plt.axis("equal")
    with writer.saving(fig, target_video_name, 100):
        for time in range(1, len(time)):
            fig.clf()
            plt.plot(
                position_of_rod1[time, 0],
                position_of_rod1[time, 1],
                color="r",
                marker="o",
                markersize=DOT_SIZE,
                linewidth=BEAM_WIDTH,
                label="rod1",
            )
            plt.plot(
                position_of_rod2[time, 0],
                position_of_rod2[time, 1],
                color=to_rgb("xkcd:bluish"),
                marker="o",
                markersize=DOT_SIZE,
                linewidth=BEAM_WIDTH,
                label="rod2",
            )

            plt.xlim([-0.25, 0.25])
            plt.ylim([-0.25, 0.25])
            writer.grab_frame()


def plot_video_xz(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])

    writer, writer_name = _make_animation_writer(manimation, fps)
    print(f"plot video xz (writer: {writer_name})")
    target_video_name = _resolve_video_name_for_writer(video_name, writer_name)
    fig = plt.figure()
    plt.axis("equal")
    with writer.saving(fig, target_video_name, 100):
        for time in range(1, len(time)):
            fig.clf()
            plt.plot(
                position_of_rod1[time, 0],
                position_of_rod1[time, 2],
                color="r",
                marker="o",
                markersize=DOT_SIZE,
                linewidth=BEAM_WIDTH,
                label="rod1",
            )
            plt.plot(
                position_of_rod2[time, 0],
                position_of_rod2[time, 2],
                color=to_rgb("xkcd:bluish"),
                marker="o",
                markersize=DOT_SIZE,
                linewidth=BEAM_WIDTH,
                label="rod2",
            )

            plt.xlim([-0.25, 0.25])
            plt.ylim([0, 0.41])
            writer.grab_frame()


def plot_video_three_rods(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    plot_params_rod3,
    video_name="video.mp4",
    fps=15,
):
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])
    position_of_rod3_all = _position_collections_from_input(plot_params_rod3)
    rod3_colors = ["g", "m", "c", "y", "k"]

    writer, writer_name = _make_animation_writer(manimation, fps)
    print(f"plot video (writer: {writer_name})")
    target_video_name = _resolve_video_name_for_writer(video_name, writer_name)
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    with writer.saving(fig, target_video_name, 100):
        for frame_id in range(1, len(time)):
            fig.clf()
            ax = plt.axes(projection="3d")
            ax.plot(
                position_of_rod1[frame_id, 0],
                position_of_rod1[frame_id, 1],
                position_of_rod1[frame_id, 2],
                color="r",
                marker="o",
                markersize=DOT_SIZE,
                linewidth=BEAM_WIDTH,
                label="rod1",
            )
            ax.plot(
                position_of_rod2[frame_id, 0],
                position_of_rod2[frame_id, 1],
                position_of_rod2[frame_id, 2],
                color="b",
                marker="o",
                markersize=DOT_SIZE,
                linewidth=BEAM_WIDTH,
                label="rod2",
            )
            for rod_idx, position_of_rod3 in enumerate(position_of_rod3_all):
                ax.plot(
                    position_of_rod3[frame_id, 0],
                    position_of_rod3[frame_id, 1],
                    position_of_rod3[frame_id, 2],
                    color=rod3_colors[rod_idx % len(rod3_colors)],
                    marker="o",
                    markersize=DOT_SIZE,
                    linewidth=BEAM_WIDTH,
                    label=f"rod3_{rod_idx + 1}",
                )

            ax.set_xlim(*ZOOM_XY_XLIM)
            ax.set_ylim(*ZOOM_XY_YLIM)
            ax.set_zlim(*ZOOM_3D_ZLIM)
            writer.grab_frame()


def plot_video_xy_three_rods(
    plot_params_rod1: dict,
    plot_params_rod2: dict,
    plot_params_rod3,
    video_name="video_xy.mp4",
    fps=15,
):
    import matplotlib.animation as manimation

    time = plot_params_rod1["time"]
    position_of_rod1 = np.array(plot_params_rod1["position"])
    position_of_rod2 = np.array(plot_params_rod2["position"])
    position_of_rod3_all = _position_collections_from_input(plot_params_rod3)
    rod3_colors = ["g", "m", "c", "y", "k"]

    writer, writer_name = _make_animation_writer(manimation, fps)
    print(f"plot video xy (writer: {writer_name})")
    target_video_name = _resolve_video_name_for_writer(video_name, writer_name)
    fig = plt.figure()
    plt.axis("equal")
    with writer.saving(fig, target_video_name, 100):
        for frame_id in range(1, len(time)):
            fig.clf()
            plt.plot(
                position_of_rod1[frame_id, 0],
                position_of_rod1[frame_id, 1],
                color="r",
                marker="o",
                markersize=DOT_SIZE,
                linewidth=BEAM_WIDTH,
                label="rod1",
            )
            plt.plot(
                position_of_rod2[frame_id, 0],
                position_of_rod2[frame_id, 1],
                color="b",
                marker="o",
                markersize=DOT_SIZE,
                linewidth=BEAM_WIDTH,
                label="rod2",
            )
            for rod_idx, position_of_rod3 in enumerate(position_of_rod3_all):
                plt.plot(
                    position_of_rod3[frame_id, 0],
                    position_of_rod3[frame_id, 1],
                    color=rod3_colors[rod_idx % len(rod3_colors)],
                    marker="o",
                    markersize=DOT_SIZE,
                    linewidth=BEAM_WIDTH,
                    label=f"rod3_{rod_idx + 1}",
                )

            plt.xlim(ZOOM_XY_XLIM)
            plt.ylim(ZOOM_XY_YLIM)
            writer.grab_frame()
