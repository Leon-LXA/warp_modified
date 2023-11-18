# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""This module contains time-integration objects for simulating
models + state forward in time.

"""

import torch
import warp as wp
from .model import ModelShapeGeometry, ModelShapeMaterials


# define new matrix dimensions
mat43 = wp.types.matrix(shape=(4, 3), dtype=float)


# # Frank & Park definition 3.20, pg 100
@wp.func
def spatial_transform_twist(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_transform_wrench(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    v = wp.quat_rotate(q, v)
    w = wp.quat_rotate(q, w) + wp.cross(p, v)

    return wp.spatial_vector(w, v)


# computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates), Frank & Park, section 8.2.3, pg 290
@wp.func
def spatial_transform_inertia(t: wp.transform, I: wp.spatial_matrix):
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.mat33(r1, r2, r3)
    S = wp.mul(wp.skew(p), R)

    T = wp.spatial_adjoint(R, S)

    return wp.mul(wp.mul(wp.transpose(T), I), T)


@wp.kernel
def eval_rigid_contacts_art(
    contact_count: wp.array(dtype=int),
    body_X_s: wp.array(dtype=wp.transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    geo: ModelShapeGeometry,
    alpha: float,
    body_f_s: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    c_body = contact_body[tid]
    c_point = contact_point[tid]
    c_shape = contact_shape[tid]
    c_dist = geo.thickness[c_shape]

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = shape_materials.ke[c_shape]
    kd = shape_materials.kd[c_shape]
    kf = shape_materials.kf[c_shape]
    mu = shape_materials.mu[c_shape]

    X_s = body_X_s[c_body]  # position of colliding body
    v_s = body_v_s[c_body]  # orientation of colliding body

    n = wp.vec3(0.0, 1.0, 0.0)

    # transform point to world space
    p = wp.transform_point(X_s, c_point) - n * c_dist  # add on 'thickness' of shape, e.g.: radius of sphere/capsule

    w = wp.spatial_top(v_s)
    v = wp.spatial_bottom(v_s)

    # contact point velocity
    dpdt = v + wp.cross(w, p)  # why is this correct? yes because v and w are in spatial frame

    # check ground contact
    c = wp.dot(n, p)  # check if we're inside the ground

    if c >= 0.0:
        return

    vn = wp.dot(n, dpdt)  # velocity component out of the ground
    vt = dpdt - n * vn  # velocity component not into the ground

    fn = compute_normal_force(c, ke, alpha)  # normal force (restitution coefficient * how far inside for ground)

    # contact damping
    fd = compute_damping_force(vn, kd, c, alpha)

    # viscous friction
    # ft = vt*kf

    # Coulomb friction (box)
    # lower = mu * (fn + fd)   # negative
    # upper = 0.0 - lower      # positive, workaround for no unary ops

    # vx = wp.clamp(wp.dot(wp.vec3(kf, 0.0, 0.0), vt), lower, upper)
    # vz = wp.clamp(wp.dot(wp.vec3(0.0, 0.0, kf), vt), lower, upper)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    ft = compute_friction_force(vt, mu, kf, fn, fd, alpha)

    f_total = n * (fn + fd) + ft
    t_total = wp.cross(p, f_total)

    wp.atomic_add(body_f_s, c_body, wp.spatial_vector(t_total, f_total))


@wp.func
def compute_normal_force(c: float, ke: float, alpha: float):
    return c * ke


@wp.func
def compute_damping_force(vn: float, kd: float, c: float, alpha: float):
    return wp.min(vn, 0.0) * kd * wp.step(c)  # * (0.0 - c)


@wp.func
def compute_friction_force(vt: wp.vec3, mu: float, kf: float, fn: float, fd: float, alpha: float):
    return wp.normalize(vt) * wp.min(kf * wp.length(vt), 0.0 - mu * (fn + fd))  # * wp.step(c)


@wp.func_replay(compute_normal_force)
def replay_compute_normal_force(c: float, ke: float, alpha: float):
    return c * ke * alpha


@wp.func_replay(compute_damping_force)
def replay_compute_damping_force(vn: float, kd: float, c: float, alpha: float):
    return wp.min(vn, 0.0) * kd * wp.step(c) * alpha


@wp.func_replay(compute_friction_force)
def replay_compute_friction_force(vt: wp.vec3, mu: float, kf: float, fn: float, fd: float, alpha: float):
    return wp.normalize(vt) * wp.min(kf * alpha * wp.length(vt), 0.0 - mu * (fn + fd))


@wp.func_grad(compute_normal_force)
def adj_compute_normal_force(c: float, ke: float, alpha: float, adj_ret: float):
    # backward
    wp.adjoint[c] += ke * adj_ret * alpha
    wp.adjoint[ke] += c * adj_ret


@wp.func_grad(compute_damping_force)
def adj_compute_damping_force(vn: float, kd: float, c: float, alpha: float, adj_ret: float):
    # forward
    var_1 = wp.min(vn, 0.0)
    var_2 = var_1 * kd * alpha
    var_3 = wp.step(c)
    var_4 = var_2 * var_3

    # backward
    adj_2 = var_3 * adj_ret
    adj_3 = var_2 * adj_ret

    adj_1 = kd * adj_2
    wp.adjoint[kd] += var_1 * adj_2
    adj_vn = 0.0
    if vn < 0.0:
        adj_vn = adj_1 * alpha
    wp.adjoint[vn] += adj_vn


@wp.func_grad(compute_friction_force)
def adj_compute_friction_force(vt: wp.vec3, mu: float, kf: float, fn: float, fd: float, alpha: float, adj_ret: wp.vec3):
    # forward
    var_0 = wp.normalize(vt)
    var_1 = wp.length(vt)
    var_2 = kf * var_1 * alpha
    var_3 = 0.0
    var_4 = fn + fd
    var_5 = mu * var_4
    var_6 = var_3 - var_5
    var_7 = wp.min(var_2, var_6)
    var_8 = var_0 * var_7

    # backward
    adj_7 = wp.dot(var_0, adj_ret)
    adj_0 = var_7 * adj_ret

    if var_2 < var_6:
        adj_2 = adj_7
        adj_6 = 0.0
    else:
        adj_6 = adj_7
        adj_2 = 0.0

    adj_3 = adj_6
    adj_5 = -adj_6

    wp.adjoint[mu] += var_4 * adj_5
    adj_4 = mu * adj_5

    wp.adjoint[fn] += adj_4
    wp.adjoint[fd] += adj_4

    wp.adjoint[kf] += var_1 * adj_2
    adj_1 = kf * adj_2 * alpha

    wp.adjoint[vt] += var_0 * adj_1  # finite checks?

    # norm = var_0
    # length = var_1
    if var_1 > 1e-6:
        invd = 1.0 / var_1
        wp.adjoint[vt] += adj_0 * invd - var_0 * wp.dot(var_0, adj_0) * invd


# compute transform across a joint
@wp.func
def jcalc_transform(type: int, axis: wp.vec3, joint_q: wp.array(dtype=float), start: int):
    # prismatic
    if type == 0:
        q = joint_q[start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    # revolute
    if type == 1:
        q = joint_q[start]
        X_jc = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(axis, q))
        return X_jc

    # ball
    if type == 2:
        qx = joint_q[start + 0]
        qy = joint_q[start + 1]
        qz = joint_q[start + 2]
        qw = joint_q[start + 3]

        X_jc = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(qx, qy, qz, qw))
        return X_jc

    # fixed
    if type == 3:
        X_jc = wp.transform_identity()
        return X_jc

    # free
    if type == 4:
        px = joint_q[start + 0]
        py = joint_q[start + 1]
        pz = joint_q[start + 2]

        qx = joint_q[start + 3]
        qy = joint_q[start + 4]
        qz = joint_q[start + 5]
        qw = joint_q[start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    # default case
    return wp.transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    axis: wp.vec3,
    X_sc: wp.transform,
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_qd: wp.array(dtype=float),
    joint_start: int,
):
    # prismatic
    if type == 0:
        S_s = spatial_transform_twist(X_sc, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), axis))
        v_j_s = S_s * joint_qd[joint_start]

        joint_S_s[joint_start] = S_s
        return v_j_s

    # revolute
    if type == 1:
        S_s = spatial_transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3(0.0, 0.0, 0.0)))
        v_j_s = S_s * joint_qd[joint_start]

        joint_S_s[joint_start] = S_s
        return v_j_s

    # ball
    if type == 2:
        w = wp.vec3(joint_qd[joint_start + 0], joint_qd[joint_start + 1], joint_qd[joint_start + 2])

        S_0 = spatial_transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        S_1 = spatial_transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        S_2 = spatial_transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

        # write motion subspace
        joint_S_s[joint_start + 0] = S_0
        joint_S_s[joint_start + 1] = S_1
        joint_S_s[joint_start + 2] = S_2

        return S_0 * w[0] + S_1 * w[1] + S_2 * w[2]

    # fixed
    if type == 3:
        return wp.spatial_vector()

    # free
    if type == 4:
        v_j_s = wp.spatial_vector(
            joint_qd[joint_start + 0],
            joint_qd[joint_start + 1],
            joint_qd[joint_start + 2],
            joint_qd[joint_start + 3],
            joint_qd[joint_start + 4],
            joint_qd[joint_start + 5],
        )

        # write motion subspace
        joint_S_s[joint_start + 0] = wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        joint_S_s[joint_start + 1] = wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        joint_S_s[joint_start + 2] = wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        joint_S_s[joint_start + 3] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        joint_S_s[joint_start + 4] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        joint_S_s[joint_start + 5] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        return v_j_s

    # default case
    return wp.spatial_vector()


# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    target_k_e: float,
    target_k_d: float,
    limit_k_e: float,
    limit_k_d: float,
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    body_f_s: wp.spatial_vector,
    tau: wp.array(dtype=float),
):
    # prismatic / revolute
    if type == 0 or type == 1:
        S_s = joint_S_s[dof_start]

        q = joint_q[coord_start]
        qd = joint_qd[dof_start]
        act = joint_act[dof_start]

        target = joint_target[coord_start]
        lower = joint_limit_lower[coord_start]
        upper = joint_limit_upper[coord_start]

        limit_f = 0.0

        # compute limit forces, damping only active when limit is violated
        if q < lower:
            limit_f = limit_k_e * (lower - q)

        if q > upper:
            limit_f = limit_k_e * (upper - q)

        damping_f = (0.0 - limit_k_d) * qd

        # total torque / force on the joint
        t = (
            0.0
            - wp.spatial_dot(S_s, body_f_s)
            - target_k_e * (q - target)
            - target_k_d * qd
            + act
            + limit_f
            + damping_f
        )

        tau[dof_start] = t

    # ball
    if type == 2:
        # elastic term.. this is proportional to the
        # imaginary part of the relative quaternion
        r_j = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # angular velocity for damping
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        for i in range(0, 3):
            S_s = joint_S_s[dof_start + i]

            w = w_j[i]
            r = r_j[i]

            tau[dof_start + i] = 0.0 - wp.spatial_dot(S_s, body_f_s) - w * target_k_d - r * target_k_e

    # free
    if type == 4:
        for i in range(0, 6):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = 0.0 - wp.spatial_dot(S_s, body_f_s)

    return 0


@wp.func
def jcalc_integrate(
    type: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    dt: float,
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    # prismatic / revolute
    if type == 0 or type == 1:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        q_new = q + (qd + qd_new) / 2.0 * dt  # moreau

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

    # free joint
    if type == 4:
        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear acceleration
        m_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])

        a_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        # angular and linear velocity
        w_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        v_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # moreau
        w_s_new = w_s + m_s * dt
        w_s_avg = (w_s + w_s_new) / 2.0
        v_s_new = v_s + a_s * dt
        v_s_avg = (v_s + v_s_new) / 2.0

        # translation of origin
        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # linear vel of origin (note q/qd switch order of linear angular elements)
        # note we are converting the body twist in the space frame (w_s, v_s) to compute center of mass velcity
        dpdt_s = v_s_avg + wp.cross(w_s_avg, p_s)

        # quat and quat derivative
        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        drdt_s = wp.mul(wp.quat(w_s_avg, 0.0), r_s) * 0.5

        # new orientation (normalized)
        p_s_new = p_s + dpdt_s * dt
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        # update transform
        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        # update joint_twist
        joint_qd_new[dof_start + 0] = w_s_new[0]
        joint_qd_new[dof_start + 1] = w_s_new[1]
        joint_qd_new[dof_start + 2] = w_s_new[2]
        joint_qd_new[dof_start + 3] = v_s_new[0]
        joint_qd_new[dof_start + 4] = v_s_new[1]
        joint_qd_new[dof_start + 5] = v_s_new[2]

    return 0


@wp.func
def compute_link_transform(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_pj: wp.array(dtype=wp.transform),
    joint_X_cm: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    body_X_sc: wp.array(dtype=wp.transform),
    body_X_sm: wp.array(dtype=wp.transform),
):
    # parent transform
    parent = joint_parent[i]

    # parent transform in spatial coordinates
    X_sp = wp.transform_identity()
    if parent >= 0:
        X_sp = body_X_sc[parent]

    type = joint_type[i]
    axis = joint_axis[i]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_jc = jcalc_transform(type, axis, joint_q, coord_start)

    X_pj = joint_X_pj[i]
    X_sc = wp.transform_multiply(X_sp, wp.transform_multiply(X_pj, X_jc))

    # compute transform of center of mass
    X_cm = joint_X_cm[i]
    X_sm = wp.transform_multiply(X_sc, X_cm)

    # store geometry transforms
    body_X_sc[i] = X_sc
    body_X_sm[i] = X_sm

    return 0


@wp.func
def compute_link_velocity(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_X_sc: wp.array(dtype=wp.transform),
    body_X_sm: wp.array(dtype=wp.transform),
    joint_X_pj: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    type = joint_type[i]
    axis = joint_axis[i]
    parent = joint_parent[i]
    dof_start = joint_qd_start[i]

    # parent transform in spatial coordinates
    X_sp = wp.transform_identity()
    if parent >= 0:
        X_sp = body_X_sc[parent]

    X_pj = joint_X_pj[i]
    X_sj = wp.transform_multiply(X_sp, X_pj)

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    v_j_s = jcalc_motion(type, axis, X_sj, joint_S_s, joint_qd, dof_start)

    # parent velocity
    v_parent_s = wp.spatial_vector()
    a_parent_s = wp.spatial_vector()

    if parent >= 0:
        v_parent_s = body_v_s[parent]
        a_parent_s = body_a_s[parent]

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + wp.spatial_cross(v_s, v_j_s)  # + self.joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = body_X_sm[i]
    I_m = body_I_m[i]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    g = gravity

    m = I_m[3, 3]

    f_g_m = wp.spatial_vector(wp.vec3(), g) * m
    f_g_s = spatial_transform_wrench(wp.transform(wp.transform_get_translation(X_sm), wp.quat_identity()), f_g_m)

    # body forces
    I_s = spatial_transform_inertia(X_sm, I_m)

    f_b_s = wp.mul(I_s, a_s) + wp.spatial_cross_dual(v_s, wp.mul(I_s, v_s))

    body_v_s[i] = v_s
    body_a_s[i] = a_s
    body_f_s[i] = f_b_s - f_g_s
    body_I_s[i] = I_s

    return 0


@wp.func
def compute_link_tau(
    offset: int,
    joint_end: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    body_ft_s: wp.array(dtype=wp.spatial_vector),
    tau: wp.array(dtype=float),
):
    # for backwards traversal
    i = joint_end - offset - 1

    type = joint_type[i]
    parent = joint_parent[i]
    dof_start = joint_qd_start[i]
    coord_start = joint_q_start[i]

    target_k_e = joint_target_ke[i]
    target_k_d = joint_target_kd[i]

    limit_k_e = joint_limit_ke[i]
    limit_k_d = joint_limit_kd[i]

    # total forces on body
    f_b_s = body_fb_s[i]
    f_t_s = body_ft_s[i]

    f_s = f_b_s + f_t_s

    # compute joint-space forces, writes out tau
    jcalc_tau(
        type,
        target_k_e,
        target_k_d,
        limit_k_e,
        limit_k_d,
        joint_S_s,
        joint_q,
        joint_qd,
        joint_act,
        joint_target,
        joint_limit_lower,
        joint_limit_upper,
        coord_start,
        dof_start,
        f_s,
        tau,
    )

    # update parent forces, todo: check that this is valid for the backwards pass
    if parent >= 0:
        wp.atomic_add(body_ft_s, parent, f_s)

    return 0


@wp.kernel
def eval_rigid_fk(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_pj: wp.array(dtype=wp.transform),
    joint_X_cm: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    body_X_sc: wp.array(dtype=wp.transform),
    body_X_sm: wp.array(dtype=wp.transform),
):
    # one thread per-articulation
    tid = wp.tid()

    start = articulation_start[tid]
    end = articulation_start[tid + 1]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_X_pj,
            joint_X_cm,
            joint_axis,
            body_X_sc,
            body_X_sm,
        )


@wp.kernel
def eval_rigid_id(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_X_sc: wp.array(dtype=wp.transform),
    body_X_sm: wp.array(dtype=wp.transform),
    joint_X_pj: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    # one thread per-articulation
    tid = wp.tid()

    start = articulation_start[tid]
    end = articulation_start[tid + 1]
    count = end - start

    # compute link velocities and coriolis forces
    for i in range(start, end):
        compute_link_velocity(
            i,
            joint_type,
            joint_parent,
            joint_qd_start,
            joint_qd,
            joint_axis,
            body_I_m,
            body_X_sc,
            body_X_sm,
            joint_X_pj,
            gravity,
            joint_S_s,
            body_I_s,
            body_v_s,
            body_f_s,
            body_a_s,
        )


@wp.kernel
def eval_rigid_tau(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    body_ft_s: wp.array(dtype=wp.spatial_vector),
    tau: wp.array(dtype=float),
):
    # one thread per-articulation
    tid = wp.tid()

    start = articulation_start[tid]
    end = articulation_start[tid + 1]
    count = end - start

    # compute joint forces
    for i in range(0, count):
        compute_link_tau(
            i,
            end,
            joint_type,
            joint_parent,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_qd,
            joint_act,
            joint_target,
            joint_target_ke,
            joint_target_kd,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_S_s,
            body_fb_s,
            body_ft_s,
            tau,
        )


@wp.kernel
def eval_rigid_integrate(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    # one thread per-articulation
    tid = wp.tid()

    type = joint_type[tid]
    coord_start = joint_q_start[tid]
    dof_start = joint_qd_start[tid]

    jcalc_integrate(type, joint_q, joint_qd, joint_qdd, coord_start, dof_start, dt, joint_q_new, joint_qd_new)


@wp.kernel
def eval_rigid_jacobian(
    articulation_start: wp.array(dtype=int),
    articulation_J_start: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    J: wp.array(dtype=float),
):
    # one thread per-articulation
    tid = wp.tid()

    joint_start = articulation_start[tid]
    joint_end = articulation_start[tid + 1]
    joint_count = joint_end - joint_start

    J_offset = articulation_J_start[tid]

    wp.spatial_jacobian(joint_S_s, joint_parent, joint_qd_start, joint_start, joint_count, J_offset, J)


@wp.kernel
def eval_rigid_mass(
    articulation_start: wp.array(dtype=int),
    articulation_M_start: wp.array(dtype=int),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    # outputs
    M: wp.array(dtype=float),
):
    # one thread per-articulation
    tid = wp.tid()

    joint_start = articulation_start[tid]
    joint_end = articulation_start[tid + 1]
    joint_count = joint_end - joint_start

    M_offset = articulation_M_start[tid]

    wp.spatial_mass(body_I_s, joint_start, joint_count, M_offset, M)


@wp.kernel
def inertial_body_pos_vel(
    articulation_start: wp.array(dtype=int),
    body_X_sc: wp.array(dtype=wp.transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    # one thread per-articulation
    tid = wp.tid()

    start = articulation_start[tid]
    end = articulation_start[tid + 1]

    for i in range(start, end):
        X_sc = body_X_sc[i]
        v_s = body_v_s[i]
        w = wp.spatial_top(v_s)
        v = wp.spatial_bottom(v_s)

        v_inertial = v + wp.cross(w, wp.transform_get_translation(X_sc))

        body_q[i] = X_sc
        body_qd[i] = wp.spatial_vector(w, v_inertial)


@wp.kernel
def eval_dense_gemm(
    m: int,
    n: int,
    p: int,
    t1: int,
    t2: int,
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    C: wp.array(dtype=float),
):
    wp.dense_gemm(m, n, p, t1, t2, A, B, C)


@wp.kernel
def eval_dense_gemm_batched(
    m: wp.array(dtype=int),
    n: wp.array(dtype=int),
    p: wp.array(dtype=int),
    t1: int,
    t2: int,
    A_start: wp.array(dtype=int),
    B_start: wp.array(dtype=int),
    C_start: wp.array(dtype=int),
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    C: wp.array(dtype=float),
):
    wp.dense_gemm_batched(m, n, p, t1, t2, A_start, B_start, C_start, A, B, C)


@wp.kernel
def eval_dense_cholesky(
    n: int, A: wp.array(dtype=float), regularization: wp.array(dtype=float), L: wp.array(dtype=float)
):
    wp.dense_chol(n, A, regularization, L)


@wp.kernel
def eval_dense_cholesky_batched(
    A_start: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    regularization: wp.array(dtype=float),
    L: wp.array(dtype=float),
):
    wp.dense_chol_batched(A_start, A_dim, A, regularization, L)


@wp.kernel
def eval_dense_subs(n: int, L: wp.array(dtype=float), b: wp.array(dtype=float), x: wp.array(dtype=float)):
    wp.dense_subs(n, L, b, x)


# helper that propagates gradients back to A, treating L as a constant / temporary variable
# allows us to reuse the Cholesky decomposition from the forward pass
@wp.kernel
def eval_dense_solve(
    n: int,
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
    x: wp.array(dtype=float),
):
    wp.dense_solve(n, A, L, b, tmp, x)


# helper that propagates gradients back to A, treating L as a constant / temporary variable
# allows us to reuse the Cholesky decomposition from the forward pass
@wp.kernel
def eval_dense_solve_batched(
    b_start: wp.array(dtype=int),
    A_start: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
    x: wp.array(dtype=float),
):
    wp.dense_solve_batched(b_start, A_start, A_dim, A, L, b, tmp, x)


@wp.kernel
def eval_dense_solve_batched_matrix(
    dof_count: int,
    b_start: wp.array(dtype=int),
    A_start: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    B: wp.array(dtype=float),
    TMP: wp.array(dtype=float),
    X: wp.array(dtype=float),
):
    tid = wp.tid()
    # Jc is transposed so vectorization helps us here
    for i in range(0, 4 * 3):  # assuming 4 contacts per articulation
        wp.dense_solve_batched(b_start, A_start, A_dim, A, L, B, TMP, X)
        b_start[tid] = b_start[tid] + dof_count


@wp.kernel
def eval_dense_add_batched(
    n: int,
    start: wp.array(dtype=int),
    a: wp.array(dtype=float),
    b: wp.array(dtype=float),
    dt: float,
    c: wp.array(dtype=float),
):
    tid = wp.tid()
    for i in range(0, n):
        c[start[tid] + i] = a[start[tid] + i] + b[start[tid] + i] * dt


def matmul_batched(batch_count, m, n, k, t1, t2, A_start, B_start, C_start, A, B, C, device):
    if device == "cpu":
        threads = batch_count
    else:
        threads = 256 * batch_count  # must match the threadblock size used in adjoint.py

    wp.launch(
        kernel=eval_dense_gemm_batched,
        dim=threads,
        inputs=[
            m,
            n,
            k,
            t1,
            t2,
            A_start,
            B_start,
            C_start,
            A,
            B,
        ],
        outputs=[C],
        device=device,
    )


@wp.func
def jcalc_integrate_q(
    type: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    dt: float,
    joint_q_mid: wp.array(dtype=float),
):
    # prismatic / revolute
    if type == 0 or type == 1:
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        q_mid = q + qd * dt

        joint_q_mid[coord_start] = q_mid

    # free joint
    if type == 4:
        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear velocity
        w_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        v_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # translation of origin
        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # linear vel of origin (note q/qd switch order of linear angular elements)
        # note we are converting the body twist in the space frame (w_s, v_s) to compute center of mass velcity
        dpdt_s = v_s + wp.cross(w_s, p_s)

        # quat and quat derivative
        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        drdt_s = wp.mul(wp.quat(w_s, 0.0), r_s) * 0.5

        # mid orientation (normalized)
        p_s_mid = p_s + dpdt_s * dt
        r_s_mid = wp.normalize(r_s + drdt_s * dt)

        # update transform
        joint_q_mid[coord_start + 0] = p_s_mid[0]
        joint_q_mid[coord_start + 1] = p_s_mid[1]
        joint_q_mid[coord_start + 2] = p_s_mid[2]

        joint_q_mid[coord_start + 3] = r_s_mid[0]
        joint_q_mid[coord_start + 4] = r_s_mid[1]
        joint_q_mid[coord_start + 5] = r_s_mid[2]
        joint_q_mid[coord_start + 6] = r_s_mid[3]

    return 0


@wp.kernel
def integrate_q_halfstep(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
):
    # one thread per-articulation
    tid = wp.tid()

    type = joint_type[tid]
    coord_start = joint_q_start[tid]
    dof_start = joint_qd_start[tid]

    jcalc_integrate_q(type, joint_q, joint_qd, coord_start, dof_start, dt / 2.0, joint_q_new)


@wp.kernel
def construct_contact_jacobian(
    J: wp.array(dtype=float),
    J_start: wp.array(dtype=int),
    Jc_start: wp.array(dtype=int),
    body_X_sc: wp.array(dtype=wp.transform),
    rigid_contact_max: int,
    articulation_count: int,
    dof_count: int,
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),
    geo: ModelShapeGeometry,
    Jc: wp.array(dtype=float),
    c_body_vec: wp.array(dtype=int),
    point_vec: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    contacts_per_articulation = rigid_contact_max / articulation_count
    foot_id = int(0)

    for i in range(0, contacts_per_articulation):  # iterate all contacts
        contact_id = tid * contacts_per_articulation - 1 + i
        c_body = contact_body[contact_id]
        c_point = contact_point[contact_id]
        c_dist = geo.thickness[contact_id]

        if c_body - tid % 3 == 0 and i % 2 == 0:  # only consider foot contacts
            X_s = body_X_sc[c_body]

            n = wp.vec3(0.0, 1.0, 0.0)
            # transform point to world space
            p = (
                wp.transform_point(X_s, c_point) - n * c_dist
            )  # add on 'thickness' of shape, e.g.: radius of sphere/capsule
            # check ground contact
            c = wp.dot(n, p)

            p_skew = wp.skew(wp.vec3(p[0], p[1], p[2]))

            if c <= 0.0:
                # Jc = J_p - skew(p)*J_r
                for j in range(0, 3):  # iterate all contact dofs
                    for k in range(0, dof_count):  # iterate all joint dofs
                        Jc[dense_J_index(Jc_start, 3, dof_count, tid, foot_id, j, k)] = (
                            J[dense_J_index(J_start, 6, dof_count, tid, c_body, j, k)]
                            - p_skew[j, 0] * J[dense_J_index(J_start, 6, dof_count, tid, c_body, 3, k)]
                            - p_skew[j, 1] * J[dense_J_index(J_start, 6, dof_count, tid, c_body, 4, k)]
                            - p_skew[j, 2] * J[dense_J_index(J_start, 6, dof_count, tid, c_body, 5, k)]
                        )

            c_body_vec[tid * 4 + foot_id] = c_body
            point_vec[tid * 4 + foot_id] = p
            foot_id += 1


@wp.func
def dense_J_index(J_start: wp.array(dtype=int), dim_count: int, dof_count: int, tid: int, i: int, j: int, k: int):
    """
    J_start: articulation start index
    dim_count: number of body/contact dims
    dof_count: number of joint dofs

    tid: articulation
    i: body/contact
    j: linear/angular velocity
    k: joint velocity
    """

    return J_start[tid] + i * dim_count + j * dof_count + k  # articulation, body/contact, dim, dof


@wp.kernel
def prox_iteration(
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    percussion: wp.array2d(dtype=wp.vec3),
    mu: float,
    prox_iter: int,
):
    tid = wp.tid()

    G = G_mat[tid]
    c = c_vec[tid]
    p = percussion[tid]

    # initialize percussions with steady state
    for i in range(4):
        p[i] = -wp.inverse(G[i, i]) * c[i]
        # overwrite percussions with steady state only in normal direction
        p[i] = wp.vec3(0.0, p[i][1], 0.0)

    # solve percussions iteratively
    for it in range(prox_iter):
        for i in range(4):
            # calculate sum(G_ij*p_j) and sum over det(G_ij)
            sum = wp.vec3(0.0, 0.0, 0.0)
            r_sum = 0.0

            for j in range(4):
                sum += G[i, j] * p[j]
                r_sum += wp.determinant(G[i, j])

            r = 1.0 / (1.0 + r_sum)  # +1 for stability

            # update percussion
            p[i] = p[i] - r * (sum + c[i])

            # projection to friction cone
            if p[i][1] < 0.0:
                p[i] = wp.vec3(0.0, 0.0, 0.0)
            # friction magnitude
            fm = wp.sqrt(p[i][0] ** 2.0 + p[i][2] ** 2.0)
            if mu * p[i][1] < fm:
                p[i] = wp.vec3(p[i][0] * mu * p[i][1] / fm, p[i][1], p[i][2] * mu * p[i][1] / fm)


@wp.kernel
def convert_G_to_matrix(
    G_start: wp.array(dtype=int),
    G_vec: wp.array(dtype=float),
    G_mat: wp.array3d(dtype=wp.mat33),
):
    tid = wp.tid()

    for i in range(4):
        for j in range(4):
            G_mat[tid, i, j] = wp.mat33(
                G_vec[dense_G_index(G_start, tid, i, j, 0, 0)],
                G_vec[dense_G_index(G_start, tid, i, j, 0, 1)],
                G_vec[dense_G_index(G_start, tid, i, j, 0, 2)],
                G_vec[dense_G_index(G_start, tid, i, j, 1, 0)],
                G_vec[dense_G_index(G_start, tid, i, j, 1, 1)],
                G_vec[dense_G_index(G_start, tid, i, j, 1, 2)],
                G_vec[dense_G_index(G_start, tid, i, j, 2, 0)],
                G_vec[dense_G_index(G_start, tid, i, j, 2, 1)],
                G_vec[dense_G_index(G_start, tid, i, j, 2, 2)],
            )


@wp.func
def dense_G_index(G_start: wp.array(dtype=int), tid: int, i: int, j: int, k: int, l: int):
    """
    tid: articulation
    i: contact 1
    j: contact 2
    k: row in 3x3 matrix
    l: column in 3x3 matrix
    """
    return G_start[tid] + i * 4 * 3 * 3 + j * 3 + k + l * 4 * 3


@wp.kernel
def convert_c_to_vector(
    c: wp.array(dtype=float),
    c_vec: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    for i in range(4):
        c_start = tid * 3 * 4 + i * 3  # each articulation has 4 contacts, each contact has 3 dimensions
        c_vec[tid, i] = wp.vec3(c[c_start], c[c_start + 1], c[c_start + 2])


@wp.kernel
def p_to_f_s(
    c_body_vec: wp.array(dtype=int),
    point_vec: wp.array(dtype=wp.vec3),
    percussion: wp.array2d(dtype=wp.vec3),
    dt: float,
    body_f_s: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    p = percussion[tid]

    for i in range(4):
        f = p[i] / dt
        t = wp.cross(point_vec[tid * 4 + i], f)
        wp.atomic_add(body_f_s, c_body_vec[tid * 4 + i], wp.spatial_vector(t, f))


##########################

###  INTEGRATOR CLASS  ###

##########################


class SemiImplicitMoreauIntegrator:
    def __init__(self):
        pass

    def simulate(self, model, state_in, state_out, dt, mu, requires_grad=True, update_mass_matrix=False, prox_iter=10):
        # integrate position with euler half a step
        wp.launch(
            kernel=integrate_q_halfstep,
            dim=model.body_count,
            inputs=[
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                state_in.joint_q,
                state_in.joint_qd,
                dt,
            ],
            outputs=[
                state_in.joint_q_mid,
            ],
            device=model.device,
        )

        # evaluate mid body transforms
        wp.launch(
            kernel=eval_rigid_fk,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,  # now, originally articulation_joint_start
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state_in.joint_q_mid,
                model.joint_X_p,  # now, originally joint_X_pj
                model.joint_X_cm,
                model.joint_axis,
            ],
            outputs=[state_in.body_X_sc, state_in.body_X_sm],
            device=model.device,
        )

        # evaluate mid joint inertias, motion vectors, and forces
        wp.launch(
            kernel=eval_rigid_id,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,  # now, originally articulation_joint_start
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state_in.joint_q_mid,
                state_in.joint_qd,
                model.joint_axis,
                model.joint_target_ke,
                model.joint_target_kd,
                model.body_I_m,
                state_in.body_X_sc,
                state_in.body_X_sm,
                model.joint_X_p,  # now, originally joint_X_pj
                model.gravity,
            ],
            outputs=[
                state_in.joint_S_s,
                state_in.body_I_s,
                state_in.body_v_s,
                state_in.body_f_s,
                state_in.body_a_s,
            ],
            device=model.device,
        )

        # eval mass matrix
        if update_mass_matrix:
            self.eval_mass_matrix(model, state_in)

        # eval_tau (tau will be h)
        wp.launch(
            kernel=eval_rigid_tau,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,  # now, originally articulation_joint_start
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state_in.joint_q_mid,
                state_in.joint_qd,
                model.joint_act,
                model.joint_target,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                model.joint_axis,
                state_in.joint_S_s,
                state_in.body_f_s,
            ],
            outputs=[state_in.body_ft_s_h, state_in.joint_tau],
            device=model.device,
        )

        # construct J_c
        # do this in eval_mass_matrix, assume no switching
        wp.launch(
            kernel=construct_contact_jacobian,
            dim=model.articulation_count,
            inputs=[
                model.J,
                model.articulation_J_start,
                model.articulation_Jc_start,
                state_in.body_X_sc,
                state_in.body_v_s,
                model.rigid_contact_max,
                model.articulation_count,
                model.joint_dof_count / model.articulation_count,
                model.rigid_contact_body0,
                model.rigid_contact_point0,
                model.shape_geo,
            ],
            outputs=[model.Jc, model.c_body_vec, model.point_vec],
            device=model.device,
        )

        # find c
        # solve for x (x = H^-1*h(tau))
        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_in.joint_tau,
                state_in.tmp_inv_m_times_h,
            ],
            outputs=[state_in.inv_m_times_h],
            device=model.device,
        )

        # compute Jc*(H^-1*h(tau))
        matmul_batched(
            model.articulation_count,
            model.articulation_Jc_rows,  # m
            1,  # n
            model.articulation_Jc_cols,  # intermediate dim
            0,
            0,
            model.articulation_Jc_start,
            model.articulation_dof_start,
            model.articulation_dof_start,
            model.Jc,
            state_in.inv_m_times_h,
            state_in.Jc_times_inv_m_times_h,
            device=model.device,
        )

        # compute Jc*qd
        matmul_batched(
            model.articulation_count,
            model.articulation_Jc_rows,  # m
            1,  # n
            model.articulation_Jc_cols,  # intermediate dim
            0,
            0,
            model.articulation_Jc_start,
            model.articulation_dof_start,
            model.articulation_dof_start,
            model.Jc,
            state_in.joint_qd,
            state_in.Jc_qd,
            device=model.device,
        )

        # compute Jc*qd + Jc*(H^-1*h(tau))
        wp.launch(
            kernel=eval_dense_add_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_Jc_rows,
                model.articulation_dof_start,
                state_in.Jc_times_inv_m_times_h,
                state_in.Jc_qd,
                dt,
            ],
            outputs=[state_in.c],
            device=model.device,
        )

        # solve for X (X = H^-1*Jc^T) can also be done in eval_mass_matrix
        wp.launch(
            kernel=eval_dense_solve_batched_matrix,
            dim=model.articulation_count,
            inputs=[
                model.joint_dof_count / model.articulation_count,
                model.articulation_Jc_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                model.J_c,
                state_out.TMP,  # tmp not needed ?
            ],
            outputs=[model.Inv_M_times_Jc_t],
            device=model.device,
        )

        # compute G = Jc*(H^-1*Jc^T)
        matmul_batched(
            model.articulation_count,
            model.articulation_Jc_rows,  # m
            model.articulation_Jc_rows,  # n
            model.articulation_Jc_cols,  # intermediate dim
            0,
            1,  # Is this correct?
            model.articulation_Jc_start,
            model.articulation_Jc_start,  # What do we need here?
            model.articulation_G_start,
            model.Jc,
            model.Inv_M_times_Jc_t,
            model.G,
            device=model.device,
        )

        # convert G and c to matrix/vector arrays
        wp.launch(
            kernel=convert_G_to_matrix,
            dim=model.articulation_count,
            inputs=[
                model.articulation_G_start,
                model.articulation_G_rows,
                model.G,
            ],
            outputs=[model.G_mat],
            device=model.device,
        )

        wp.launch(
            kernel=convert_c_to_vector,
            dim=model.articulation_count,
            inputs=[
                state_in.c,
            ],
            outputs=[state_in.c_vec],
            device=model.device,
        )

        # prox iteration
        wp.launch(
            kernel=prox_iteration,
            dim=model.articulation_count,
            inputs=[
                model.G_mat,
                state_in.c_vec,
                mu,
                prox_iter,
            ],
            outputs=[state_in.percussion],
            device=model.device,
        )

        # map p to body forces
        wp.launch(
            kernel=p_to_f_s,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                model.rigid_contact_max,
                model.rigid_contact_body0,
                model.rigid_contact_point0,
                state_in.percussion,
            ],
            outputs=[state_in.body_f_s],
            device=model.device,
        )

        # recompute tau with contact forces
        wp.launch(
            kernel=eval_rigid_tau,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,  # now, originally articulation_joint_start
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state_in.joint_q_mid,
                state_in.joint_qd,
                model.joint_act,
                model.joint_target,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                model.joint_axis,
                state_in.joint_S_s,
                state_in.body_f_s,
            ],
            outputs=[state_out.body_ft_s, state_out.joint_tau],
            device=model.device,
        )

        # solve for qdd (qdd = M^-1*tau)
        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                model.articulation_dof_start,
                model.articulation_H_start,
                model.articulation_H_rows,
                model.H,
                model.L,
                state_out.joint_tau,
                state_out.tmp,
            ],
            outputs=[state_out.joint_qdd],
            device=model.device,
        )

        # integrate
        wp.launch(
            kernel=eval_rigid_integrate,
            dim=model.body_count,
            inputs=[
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                state_in.joint_q,
                state_in.joint_qd,
                state_out.joint_qdd,
                dt,
            ],
            outputs=[
                state_out.joint_q,
                state_out.joint_qd,
            ],
            device=model.device,
        )

        # evaluate final body transforms
        wp.launch(
            kernel=eval_rigid_fk,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,  # now, originally articulation_joint_start
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state_out.joint_q,
                model.joint_X_p,  # now, originally joint_X_pj
                model.joint_X_cm,
                model.joint_axis,
            ],
            outputs=[state_out.body_X_sc, state_out.body_X_sm],
            device=model.device,
        )

        # evaluate final joint inertias, motion vectors, and forces
        wp.launch(
            kernel=eval_rigid_id,
            dim=model.articulation_count,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state_out.joint_q,
                state_out.joint_qd,
                model.joint_axis,
                model.joint_target_ke,
                model.joint_target_kd,
                model.body_I_m,
                state_out.body_X_sc,
                state_out.body_X_sm,
                model.joint_X_p,  # now, originally joint_X_pj
                model.gravity,
            ],
            outputs=[
                state_out.joint_S_s,
                state_out.body_I_s,
                state_out.body_v_s,
                state_out.body_f_s,
                state_out.body_a_s,
            ],
            device=model.device,
        )

        # body position and velocity in inertial frame
        wp.launch(
            kernel=inertial_body_pos_vel,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                state_out.body_X_sc,
                state_out.body_v_s,
            ],
            outputs=[
                state_out.body_q,
                state_out.body_qd,
            ],
        )

        return state_out

    def eval_mass_matrix(self, model, state_in):
        # build J
        wp.launch(
            kernel=eval_rigid_jacobian,
            dim=model.articulation_count,
            inputs=[
                # inputs
                model.articulation_start,  # now, originally articulation_joint_start
                model.articulation_J_start,
                model.joint_parent,
                model.joint_qd_start,
                state_in.joint_S_s,
            ],
            outputs=[model.J],
            device=model.device,
        )

        # build M
        wp.launch(
            kernel=eval_rigid_mass,
            dim=model.articulation_count,
            inputs=[
                # inputs
                model.articulation_start,  # now, originally articulation_joint_start
                model.articulation_M_start,
                state_in.body_I_s,
            ],
            outputs=[model.M],
            device=model.device,
        )

        # form P = M*J
        matmul_batched(
            model.articulation_count,
            model.articulation_M_rows,
            model.articulation_J_cols,
            model.articulation_J_rows,
            0,
            0,
            model.articulation_M_start,
            model.articulation_J_start,
            model.articulation_J_start,  # P start is the same as J start since it has the same dims as J
            model.M,
            model.J,
            model.P,
            device=model.device,
        )

        # form H = J^T*P
        matmul_batched(
            model.articulation_count,
            model.articulation_J_cols,
            model.articulation_J_cols,
            model.articulation_J_rows,  # P rows is the same as J rows
            1,
            0,
            model.articulation_J_start,
            model.articulation_J_start,  # P start is the same as J start since it has the same dims as J
            model.articulation_H_start,
            model.J,
            model.P,
            model.H,
            device=model.device,
        )

        # compute decomposition
        wp.launch(
            kernel=eval_dense_cholesky_batched,
            dim=model.articulation_count,
            inputs=[model.articulation_H_start, model.articulation_H_rows, model.H, model.joint_armature],
            outputs=[model.L],
            device=model.device,
        )
