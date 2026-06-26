"""IMU fusion calibration: Full-LM vs Deep-LM (implicit bilevel optimization).

This example solves the same synthetic IMU fusion task two ways:
- Full-LM: jointly optimize poses, velocities, and a global IMU bias in one
  nonlinear least-squares problem.
- Deep-LM: optimize poses and velocities in an inner solve, while learning the
  same global IMU bias in an outer loop by differentiating through the solver.

Both methods should recover similar bias values and trajectories.
"""

import logging
import time

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as onp
import viser

try:
    import optax
except ImportError as e:
    raise ImportError(
        "This example requires optax. Install example dependencies with "
        '`pip install "jaxls[examples]"`.'
    ) from e

logging.getLogger("jaxls").setLevel(logging.ERROR)
try:
    from loguru import logger

    logger.disable("jaxls")
except ImportError:
    pass

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


class VelocityVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(3)):
    """3D velocity in world frame."""


class BiasVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(6)):
    """Global IMU bias [accel_bias(3), gyro_bias(3)]."""


def trajectory_state(t: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Smooth 3D trajectory and Euler angles at time t."""
    theta = 2.0 * jnp.pi * t / 8.0
    dtheta = 2.0 * jnp.pi / 8.0

    position = jnp.array(
        [
            3.0 * jnp.sin(theta),
            2.0 * jnp.sin(theta) * jnp.cos(theta),
            2.0 + jnp.sin(2.0 * theta),
        ]
    )

    vx = 3.0 * jnp.cos(theta) * dtheta
    vy = 2.0 * jnp.cos(2.0 * theta) * dtheta
    vz = 2.0 * jnp.cos(2.0 * theta) * dtheta

    roll = 0.25 * jnp.sin(2.0 * theta)
    yaw = jnp.arctan2(vy, vx)
    pitch = jnp.arctan2(vz, jnp.sqrt(vx**2 + vy**2))
    euler = jnp.array([roll, pitch, yaw])
    return position, euler


def generate_trajectory_at_times(
    times: jax.Array,
) -> tuple[jaxlie.SE3, jax.Array, jax.Array, jax.Array]:
    """Generate poses, velocities, accelerations, and body-frame angular rates."""

    velocity_fn = jax.jacfwd(lambda t: trajectory_state(t)[0])
    acceleration_fn = jax.jacfwd(velocity_fn)
    euler_dot_fn = jax.jacfwd(lambda t: trajectory_state(t)[1])

    def compute_state(t: jax.Array) -> tuple[jaxlie.SE3, jax.Array, jax.Array, jax.Array]:
        position, euler = trajectory_state(t)
        velocity = velocity_fn(t)
        acceleration = acceleration_fn(t)

        roll, pitch, yaw = euler
        roll_dot, pitch_dot, yaw_dot = euler_dot_fn(t)

        omega_body = jnp.array(
            [
                roll_dot - yaw_dot * jnp.sin(pitch),
                pitch_dot * jnp.cos(roll)
                + yaw_dot * jnp.cos(pitch) * jnp.sin(roll),
                -pitch_dot * jnp.sin(roll)
                + yaw_dot * jnp.cos(pitch) * jnp.cos(roll),
            ]
        )

        rotation = (
            jaxlie.SO3.from_z_radians(yaw)
            @ jaxlie.SO3.from_y_radians(pitch)
            @ jaxlie.SO3.from_x_radians(roll)
        )
        pose = jaxlie.SE3.from_rotation_and_translation(rotation, position)
        return pose, velocity, acceleration, omega_body

    return jax.vmap(compute_state)(times)


def generate_imu_measurements(
    key: jax.Array,
    true_bias: jax.Array,
    gravity: jax.Array,
    n_intervals: int,
    keyframe_dt: float,
    imu_dt: float,
    imu_measurements_per_interval: int,
) -> tuple[jax.Array, jax.Array]:
    """Generate synthetic accelerometer/gyroscope sequences with bias + noise."""
    interval_starts = jnp.arange(n_intervals) * keyframe_dt
    imu_offsets = (jnp.arange(imu_measurements_per_interval) + 0.5) * imu_dt
    times_imu = (interval_starts[:, None] + imu_offsets[None, :]).reshape(-1)

    poses_imu, _, acc_world, omega_body = generate_trajectory_at_times(times_imu)
    accel_body = jax.vmap(lambda R, a: R.inverse() @ a)(
        poses_imu.rotation(), acc_world - gravity[None, :]
    ).reshape(n_intervals, imu_measurements_per_interval, 3)
    gyro_body = omega_body.reshape(n_intervals, imu_measurements_per_interval, 3)

    key_accel, key_gyro = jax.random.split(key)
    accel_noise = 0.01 * jax.random.normal(key_accel, accel_body.shape)
    gyro_noise = 0.001 * jax.random.normal(key_gyro, gyro_body.shape)

    accel_meas = accel_body + true_bias[None, None, :3] + accel_noise
    gyro_meas = gyro_body + true_bias[None, None, 3:] + gyro_noise
    return accel_meas, gyro_meas


def preintegrate_imu(
    accel_meas: jax.Array,
    gyro_meas: jax.Array,
    accel_bias: jax.Array,
    gyro_bias: jax.Array,
    imu_dt: float,
) -> tuple[jaxlie.SO3, jax.Array, jax.Array]:
    """Preintegrate IMU measurements over one keyframe interval."""
    accel_corrected = accel_meas - accel_bias[None, :]
    gyro_corrected = gyro_meas - gyro_bias[None, :]

    def step(
        carry: tuple[jaxlie.SO3, jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jaxlie.SO3, jax.Array, jax.Array], None]:
        delta_R, delta_v, delta_p = carry
        accel, gyro = inputs

        delta_R_new = delta_R @ jaxlie.SO3.exp(gyro * imu_dt)
        accel_rotated = delta_R @ accel
        delta_p_new = delta_p + delta_v * imu_dt + 0.5 * accel_rotated * imu_dt**2
        delta_v_new = delta_v + accel_rotated * imu_dt
        return (delta_R_new, delta_v_new, delta_p_new), None

    (delta_R, delta_v, delta_p), _ = jax.lax.scan(
        step,
        (jaxlie.SO3.identity(), jnp.zeros(3), jnp.zeros(3)),
        (accel_corrected, gyro_corrected),
    )
    return delta_R, delta_v, delta_p


def imu_transition_residual(
    T_i: jaxlie.SE3,
    v_i: jax.Array,
    T_j: jaxlie.SE3,
    v_j: jax.Array,
    bias: jax.Array,
    accel_meas: jax.Array,
    gyro_meas: jax.Array,
    keyframe_dt: float,
    imu_dt: float,
    gravity: jax.Array,
) -> jax.Array:
    """Residual enforcing IMU-driven transition consistency."""
    delta_R, delta_v, delta_p = preintegrate_imu(
        accel_meas=accel_meas,
        gyro_meas=gyro_meas,
        accel_bias=bias[:3],
        gyro_bias=bias[3:],
        imu_dt=imu_dt,
    )

    R_i, p_i = T_i.rotation(), T_i.translation()
    R_j, p_j = T_j.rotation(), T_j.translation()

    p_j_pred = p_i + v_i * keyframe_dt + 0.5 * gravity * keyframe_dt**2 + R_i @ delta_p
    v_j_pred = v_i + gravity * keyframe_dt + R_i @ delta_v
    R_j_pred = R_i @ delta_R

    r_p = (p_j - p_j_pred) * 8.0
    r_v = (v_j - v_j_pred) * 5.0
    r_R = (R_j_pred.inverse() @ R_j).log() * 30.0
    return jnp.concatenate([r_p, r_v, r_R])


@jaxls.Cost.factory
def imu_cost_full(
    vals: jaxls.VarValues,
    pose_i: jaxls.SE3Var,
    vel_i: VelocityVar,
    pose_j: jaxls.SE3Var,
    vel_j: VelocityVar,
    bias_var: BiasVar,
    accel_meas: jax.Array,
    gyro_meas: jax.Array,
    keyframe_dt: float,
    imu_dt: float,
    gravity: jax.Array,
) -> jax.Array:
    return imu_transition_residual(
        vals[pose_i],
        vals[vel_i],
        vals[pose_j],
        vals[vel_j],
        vals[bias_var],
        accel_meas,
        gyro_meas,
        keyframe_dt,
        imu_dt,
        gravity,
    )


@jaxls.Cost.factory
def imu_cost_deep(
    vals: jaxls.VarValues,
    pose_i: jaxls.SE3Var,
    vel_i: VelocityVar,
    pose_j: jaxls.SE3Var,
    vel_j: VelocityVar,
    bias_batched: jax.Array,
    accel_meas: jax.Array,
    gyro_meas: jax.Array,
    keyframe_dt: float,
    imu_dt: float,
    gravity: jax.Array,
) -> jax.Array:
    return imu_transition_residual(
        vals[pose_i],
        vals[vel_i],
        vals[pose_j],
        vals[vel_j],
        bias_batched,
        accel_meas,
        gyro_meas,
        keyframe_dt,
        imu_dt,
        gravity,
    )


@jaxls.Cost.factory
def pose_prior_cost(
    vals: jaxls.VarValues,
    var: jaxls.SE3Var,
    target: jaxlie.SE3,
) -> jax.Array:
    e = (vals[var].inverse() @ target).log()
    return jnp.concatenate([e[:3] * 30.0, e[3:] * 50.0])


@jaxls.Cost.factory
def position_prior_cost(
    vals: jaxls.VarValues,
    var: jaxls.SE3Var,
    target_position: jax.Array,
) -> jax.Array:
    return (vals[var].translation() - target_position) * 15.0


@jaxls.Cost.factory
def velocity_prior_cost(
    vals: jaxls.VarValues,
    var: VelocityVar,
    target_velocity: jax.Array,
) -> jax.Array:
    return (vals[var] - target_velocity) * 20.0


@jaxls.Cost.factory
def bias_prior_cost(
    vals: jaxls.VarValues,
    var: BiasVar,
    target_bias: jax.Array,
) -> jax.Array:
    return (vals[var] - target_bias) * 0.1


@jax.jit
def dead_reckon_trajectory(
    initial_pose: jaxlie.SE3,
    initial_vel: jax.Array,
    accel_meas: jax.Array,
    gyro_meas: jax.Array,
    keyframe_dt: float,
    imu_dt: float,
    gravity: jax.Array,
) -> tuple[jaxlie.SE3, jax.Array]:
    """Dead-reckon with zero IMU bias for initialization."""
    zero_bias = jnp.zeros(6)

    def step(
        carry: tuple[jaxlie.SE3, jax.Array],
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jaxlie.SE3, jax.Array], tuple[jax.Array, jax.Array]]:
        pose, vel = carry
        accel, gyro = inputs
        delta_R, delta_v, delta_p = preintegrate_imu(
            accel_meas=accel,
            gyro_meas=gyro,
            accel_bias=zero_bias[:3],
            gyro_bias=zero_bias[3:],
            imu_dt=imu_dt,
        )

        R_i, p_i = pose.rotation(), pose.translation()
        p_next = p_i + vel * keyframe_dt + 0.5 * gravity * keyframe_dt**2 + R_i @ delta_p
        v_next = vel + gravity * keyframe_dt + R_i @ delta_v
        next_pose = jaxlie.SE3.from_rotation_and_translation(R_i @ delta_R, p_next)
        return (next_pose, v_next), (next_pose.wxyz_xyz, v_next)

    _, (pose_wxyz_xyz, velocities) = jax.lax.scan(
        step, (initial_pose, initial_vel), (accel_meas, gyro_meas)
    )
    poses = jaxlie.SE3(
        wxyz_xyz=jnp.concatenate([initial_pose.wxyz_xyz[None], pose_wxyz_xyz], axis=0)
    )
    velocities = jnp.concatenate([initial_vel[None], velocities], axis=0)
    return poses, velocities


def add_trajectory(
    server: viser.ViserServer,
    name: str,
    positions: onp.ndarray,
    color: tuple[int, int, int],
    line_width: float,
    dashed: bool = False,
) -> None:
    segments = onp.stack([positions[:-1], positions[1:]], axis=1)
    if dashed:
        segments = segments[::2]
    server.scene.add_line_segments(
        f"{name}/line",
        points=segments,
        colors=color,
        line_width=line_width,
    )
    for i, point in enumerate(positions):
        server.scene.add_icosphere(
            f"{name}/node_{i}",
            radius=0.03,
            position=(float(point[0]), float(point[1]), float(point[2])),
            color=color,
        )


def format_vec(vec: jax.Array) -> str:
    return onp.array2string(onp.array(vec), precision=4, suppress_small=True)


def main() -> None:
    print("IMU Fusion Calibration: Full-LM vs Deep-LM")
    print("=" * 60)

    n_keyframes = 14
    keyframe_dt = 0.4
    imu_measurements_per_interval = 8
    imu_dt = keyframe_dt / imu_measurements_per_interval
    n_intervals = n_keyframes - 1
    gps_interval = 4
    outer_iterations = 50
    outer_lr = 0.02

    gravity = jnp.array([0.0, 0.0, -9.81])
    true_bias = jnp.array([0.25, -0.15, 0.12, 0.02, -0.01, 0.015])

    # Generate synthetic trajectory and IMU measurements.
    keyframe_times = jnp.arange(n_keyframes) * keyframe_dt
    true_poses, true_velocities, _, _ = generate_trajectory_at_times(keyframe_times)
    accel_measurements, gyro_measurements = generate_imu_measurements(
        key=jax.random.PRNGKey(0),
        true_bias=true_bias,
        gravity=gravity,
        n_intervals=n_intervals,
        keyframe_dt=keyframe_dt,
        imu_dt=imu_dt,
        imu_measurements_per_interval=imu_measurements_per_interval,
    )

    # Sparse position fixes (GPS-like).
    gps_indices = jnp.arange(0, n_keyframes, gps_interval)
    gps_positions = true_poses.translation()[gps_indices]

    # Initialization from dead reckoning.
    initial_pose = jaxlie.SE3(wxyz_xyz=true_poses.wxyz_xyz[0])
    initial_vel = true_velocities[0]
    initial_poses, initial_velocities = dead_reckon_trajectory(
        initial_pose=initial_pose,
        initial_vel=initial_vel,
        accel_meas=accel_measurements,
        gyro_meas=gyro_measurements,
        keyframe_dt=keyframe_dt,
        imu_dt=imu_dt,
        gravity=gravity,
    )

    pose_vars = jaxls.SE3Var(id=jnp.arange(n_keyframes))
    vel_vars = VelocityVar(id=jnp.arange(n_keyframes))
    bias_var = BiasVar(id=0)
    imu_i_ids = jnp.arange(n_intervals)
    imu_j_ids = imu_i_ids + 1
    gravity_batched = jnp.broadcast_to(gravity[None, :], (n_intervals, 3))

    # -------------------------------------------------------------------------
    # Full-LM: joint optimization over states + bias
    # -------------------------------------------------------------------------
    full_costs: list[jaxls.Cost] = [
        pose_prior_cost(jaxls.SE3Var(id=0), initial_pose),
        velocity_prior_cost(VelocityVar(id=0), initial_vel),
        position_prior_cost(jaxls.SE3Var(id=gps_indices), gps_positions),
        bias_prior_cost(bias_var, jnp.zeros(6)),
        imu_cost_full(
            jaxls.SE3Var(id=imu_i_ids),
            VelocityVar(id=imu_i_ids),
            jaxls.SE3Var(id=imu_j_ids),
            VelocityVar(id=imu_j_ids),
            bias_var,
            accel_measurements,
            gyro_measurements,
            keyframe_dt,
            imu_dt,
            gravity_batched,
        ),
    ]

    full_problem = jaxls.LeastSquaresProblem(
        costs=full_costs,
        variables=[pose_vars, vel_vars, bias_var],
    ).analyze()
    full_initial_vals = jaxls.VarValues.make(
        [
            pose_vars.with_value(initial_poses),
            vel_vars.with_value(initial_velocities),
            bias_var.with_value(jnp.zeros(6)),
        ]
    )

    t0 = time.perf_counter()
    full_solution = full_problem.solve(
        initial_vals=full_initial_vals,
        linear_solver="dense_cholesky",
        termination=jaxls.TerminationConfig(max_iterations=60, cost_tolerance=1e-7),
        verbose=False,
    )
    jax.block_until_ready(full_solution)
    full_time = time.perf_counter() - t0
    full_bias = full_solution[bias_var]
    full_poses = full_solution[pose_vars]

    # -------------------------------------------------------------------------
    # Deep-LM: outer optimization on bias, inner solve on states
    # -------------------------------------------------------------------------
    inner_initial_vals = jaxls.VarValues.make(
        [
            pose_vars.with_value(initial_poses),
            vel_vars.with_value(initial_velocities),
        ]
    )

    def make_deep_problem(bias: jax.Array) -> jaxls.AnalyzedLeastSquaresProblem:
        # jaxls cost batching requires leading batch axis compatible with IMU edge
        # batching, so we tile the global bias over intervals.
        bias_batched = jnp.broadcast_to(bias[None, :], (n_intervals, 6))
        deep_costs: list[jaxls.Cost] = [
            pose_prior_cost(jaxls.SE3Var(id=0), initial_pose),
            velocity_prior_cost(VelocityVar(id=0), initial_vel),
            position_prior_cost(jaxls.SE3Var(id=gps_indices), gps_positions),
            imu_cost_deep(
                jaxls.SE3Var(id=imu_i_ids),
                VelocityVar(id=imu_i_ids),
                jaxls.SE3Var(id=imu_j_ids),
                VelocityVar(id=imu_j_ids),
                bias_batched,
                accel_measurements,
                gyro_measurements,
                keyframe_dt,
                imu_dt,
                gravity_batched,
            ),
        ]
        return jaxls.LeastSquaresProblem(
            costs=deep_costs,
            variables=[pose_vars, vel_vars],
        ).analyze()

    def deep_outer_loss(bias: jax.Array) -> jax.Array:
        problem = make_deep_problem(bias)
        solution = problem.solve_differentiable(
            initial_vals=inner_initial_vals,
            linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(max_iterations=50, cost_tolerance=1e-7),
            verbose=False,
        )
        residual = problem.compute_residual_vector(solution)
        # Match the Full-LM bias prior in the reduced objective.
        return jnp.sum(residual**2) + jnp.sum((bias * 0.1) ** 2)

    deep_loss_and_grad = jax.jit(jax.value_and_grad(deep_outer_loss))
    outer_opt = optax.adam(outer_lr)
    deep_bias = jnp.zeros(6)
    outer_state = outer_opt.init(deep_bias)
    deep_losses: list[float] = []

    t0 = time.perf_counter()
    for i in range(outer_iterations):
        loss, grad = deep_loss_and_grad(deep_bias)
        updates, outer_state = outer_opt.update(grad, outer_state, deep_bias)
        deep_bias = optax.apply_updates(deep_bias, updates)
        deep_losses.append(float(loss))

        if i % 10 == 0 or i == outer_iterations - 1:
            print(
                f"  Deep iter {i:3d}: outer_loss={float(loss):.6e}, "
                f"||bias - full||_2={float(jnp.linalg.norm(deep_bias - full_bias)):.6e}"
            )

    jax.block_until_ready(deep_bias)
    deep_time = time.perf_counter() - t0

    deep_problem_final = make_deep_problem(deep_bias)
    deep_solution = deep_problem_final.solve_differentiable(
        initial_vals=inner_initial_vals,
        linear_solver="dense_cholesky",
        termination=jaxls.TerminationConfig(max_iterations=50, cost_tolerance=1e-7),
        verbose=False,
    )
    deep_poses = deep_solution[pose_vars]

    # -------------------------------------------------------------------------
    # Comparison metrics
    # -------------------------------------------------------------------------
    true_positions = true_poses.translation()
    init_positions = initial_poses.translation()
    full_positions = full_poses.translation()
    deep_positions = deep_poses.translation()

    init_err = jnp.linalg.norm(init_positions - true_positions, axis=-1)
    full_err = jnp.linalg.norm(full_positions - true_positions, axis=-1)
    deep_err = jnp.linalg.norm(deep_positions - true_positions, axis=-1)

    print()
    print("Bias comparison:")
    print(f"  True bias:             {format_vec(true_bias)}")
    print(
        f"  Full-LM bias:          {format_vec(full_bias)} "
        f"(abs err {format_vec(jnp.abs(full_bias - true_bias))})"
    )
    print(
        f"  Deep-LM bias:          {format_vec(deep_bias)} "
        f"(abs err {format_vec(jnp.abs(deep_bias - true_bias))})"
    )
    print(
        f"  Full vs Deep abs diff: {format_vec(jnp.abs(full_bias - deep_bias))}"
    )
    print()
    print("Trajectory position error (meters):")
    print(
        f"  Dead reckoning: mean={float(jnp.mean(init_err)):.4f}, "
        f"max={float(jnp.max(init_err)):.4f}"
    )
    print(
        f"  Full-LM:       mean={float(jnp.mean(full_err)):.4f}, "
        f"max={float(jnp.max(full_err)):.4f}"
    )
    print(
        f"  Deep-LM:       mean={float(jnp.mean(deep_err)):.4f}, "
        f"max={float(jnp.max(deep_err)):.4f}"
    )
    print()
    print(f"Timing: Full-LM solve={full_time:.3f}s, Deep-LM outer={deep_time:.3f}s")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    print()
    print("Starting Viser visualization server...")
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=10.0,
        height=10.0,
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
    )

    add_trajectory(
        server,
        "/trajectory/ground_truth",
        onp.array(true_positions),
        color=(40, 170, 60),
        line_width=4.0,
    )
    add_trajectory(
        server,
        "/trajectory/dead_reckoning",
        onp.array(init_positions),
        color=(220, 80, 80),
        line_width=2.0,
        dashed=True,
    )
    add_trajectory(
        server,
        "/trajectory/full_lm",
        onp.array(full_positions),
        color=(70, 130, 220),
        line_width=3.0,
    )
    add_trajectory(
        server,
        "/trajectory/deep_lm",
        onp.array(deep_positions),
        color=(230, 160, 50),
        line_width=3.0,
    )

    for i, idx in enumerate(onp.array(gps_indices)):
        p = onp.array(gps_positions[i])
        server.scene.add_icosphere(
            f"/gps/point_{int(idx)}",
            radius=0.04,
            position=(float(p[0]), float(p[1]), float(p[2])),
            color=(240, 220, 70),
        )

    summary_md = (
        "### IMU Calibration Comparison\n"
        f"- True bias: `{format_vec(true_bias)}`\n"
        f"- Full-LM bias: `{format_vec(full_bias)}`\n"
        f"- Deep-LM bias: `{format_vec(deep_bias)}`\n"
        f"- Full vs Deep abs diff: `{format_vec(jnp.abs(full_bias - deep_bias))}`\n"
        f"- Position RMSE (Full): "
        f"`{float(jnp.sqrt(jnp.mean(full_err**2))):.4f} m`\n"
        f"- Position RMSE (Deep): "
        f"`{float(jnp.sqrt(jnp.mean(deep_err**2))):.4f} m`"
    )
    server.gui.add_markdown(summary_md)

    if go is not None:
        fig = go.Figure()
        fig.add_scatter(
            x=onp.arange(len(deep_losses)),
            y=onp.array(deep_losses),
            mode="lines",
            line={"color": "#e67e22", "width": 2},
            name="Deep-LM outer loss",
        )
        fig.update_layout(
            title="Deep-LM Outer Loss",
            xaxis_title="Outer iteration",
            yaxis_title="Reduced objective",
            yaxis_type="log",
            showlegend=False,
            height=300,
            margin={"l": 50, "r": 20, "t": 45, "b": 45},
        )
        server.gui.add_plotly(fig, aspect=1.6)
    else:
        server.gui.add_markdown("`plotly` is unavailable; skipping loss plot.")

    print(
        f"Visualization server running at http://{server.get_host()}:{server.get_port()}"
    )
    server.sleep_forever()


if __name__ == "__main__":
    main()
