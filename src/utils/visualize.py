import os
import time
import subprocess

import cv2
import numpy as np
import viser
import viser.transforms as vtf

from .joints import (
    HAND_KEYPOINT_NAMES,
    HAND_PARENT_INDICES,
    HAND_KEYPOINT_INDEX,
)


NUM_HAND_JOINTS = len(HAND_KEYPOINT_NAMES)

LEFT_HAND_COLOR = (100, 180, 255)
RIGHT_HAND_COLOR = (255, 150, 100)
GAZE_COLOR = (0, 255, 0)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def _rot_to_wxyz(rotation_matrix):
    return vtf.SO3.from_matrix(rotation_matrix).wxyz


def _save_rgb_video_with_ffmpeg(frames_rgb, save_path, fps=30):
    frames_rgb = np.asarray(frames_rgb)
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected frames shape [T,H,W,3], got {frames_rgb.shape}")

    if save_path is None:
        return

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    t, h, w, _ = frames_rgb.shape
    if t == 0:
        raise ValueError('No frames to save.')

    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        save_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _, stderr = proc.communicate(input=frames_rgb.astype(np.uint8).tobytes())
        if proc.returncode != 0:
            err = stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f'ffmpeg failed with code {proc.returncode}: {err}')
    finally:
        if proc.poll() is None:
            proc.kill()


# -----------------------------------------------------------------------------
# 3D interactive keypoint visualization (Viser)
# -----------------------------------------------------------------------------

def _build_skeleton_lines(positions, color):
    lines = []
    for i in range(1, min(len(positions), NUM_HAND_JOINTS)):
        parent = HAND_PARENT_INDICES[i]
        if parent < 0 or parent >= len(positions):
            continue
        lines.append([positions[parent], positions[i]])

    if not lines:
        return np.zeros((0, 2, 3)), np.zeros((0, 2, 3), dtype=np.uint8)

    lines = np.asarray(lines)
    colors = np.full((len(lines), 2, 3), color, dtype=np.uint8)
    return lines, colors


def _extract_hand_arrays(hand_dict):
    positions = np.zeros((NUM_HAND_JOINTS, 3), dtype=np.float32)
    rotations = np.tile(np.eye(3, dtype=np.float32), (NUM_HAND_JOINTS, 1, 1))
    mask = np.zeros(NUM_HAND_JOINTS, dtype=bool)

    for name, tf in hand_dict.items():
        idx = HAND_KEYPOINT_INDEX.get(name)
        if idx is None:
            continue
        positions[idx] = tf[:3, 3]
        rotations[idx] = tf[:3, :3]
        mask[idx] = True

    return positions, rotations, mask


def _render_hand_3d(server, prefix, hand_frame_dict, color):
    positions, rotations, mask = _extract_hand_arrays(hand_frame_dict)

    if mask[0]:
        server.scene.add_frame(
            f'{prefix}/wrist',
            wxyz=_rot_to_wxyz(rotations[0]),
            position=tuple(positions[0]),
            axes_length=0.04,
            axes_radius=0.002,
        )

    for i in range(1, NUM_HAND_JOINTS):
        if not mask[i]:
            continue
        server.scene.add_icosphere(
            f'{prefix}/joint/{HAND_KEYPOINT_NAMES[i]}',
            radius=0.005,
            color=color,
            position=tuple(positions[i]),
        )

    lines, colors = _build_skeleton_lines(positions, color)
    if len(lines) > 0:
        server.scene.add_line_segments(
            f'{prefix}/skeleton',
            points=lines,
            colors=colors,
            line_width=2.0,
        )


def _render_frame_3d(server, cam_img, cam_tf, left_frame, right_frame):
    h, w = cam_img.shape[:2]

    server.scene.add_frame(
        '/camera',
        wxyz=_rot_to_wxyz(cam_tf[:3, :3]),
        position=tuple(cam_tf[:3, 3]),
        axes_length=0.1,
        axes_radius=0.005,
    )

    r_frustum_to_parent = np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    server.scene.add_camera_frustum(
        '/camera/frustum',
        fov=np.pi / 3,
        aspect=w / h,
        scale=0.3,
        image=cam_img,
        wxyz=_rot_to_wxyz(r_frustum_to_parent),
    )

    if left_frame:
        _render_hand_3d(server, '/left', left_frame, LEFT_HAND_COLOR)
    if right_frame:
        _render_hand_3d(server, '/right', right_frame, RIGHT_HAND_COLOR)



def vis_3d_keypoint(cam_img, cam_ext, hand, gaze=None, mask=None, port=8080):
    """Visualize global camera/hand transforms in 3D.

    Args:
        cam_img: np.ndarray [T, H, W, 3]
        cam_ext: np.ndarray [T, 4, 4], camera pose in global frame
        hand: {
            'left': {'joint_name': np.ndarray [T,4,4], ...} or None,
            'right': {'joint_name': np.ndarray [T,4,4], ...} or None,
        }
        gaze: np.ndarray [T,2] image pixel coordinates (optional)
        mask: {'left': np.ndarray [T], 'right': np.ndarray [T], 'gaze': np.ndarray [T]} (optional)
    """
    cam_tfs = np.asarray(cam_ext, dtype=np.float32)
    left_data = hand.get('left') if hand is not None else None
    right_data = hand.get('right') if hand is not None else None
    total_frames = cam_img.shape[0]

    if cam_tfs.shape[0] != total_frames:
        raise ValueError('cam_img and cam_ext frame count mismatch')

    if mask is None:
        left_mask = np.ones(total_frames, dtype=bool)
        right_mask = np.ones(total_frames, dtype=bool)
        gaze_mask = np.ones(total_frames, dtype=bool)
    else:
        left_mask = np.asarray(mask.get('left', np.ones(total_frames)), dtype=bool)
        right_mask = np.asarray(mask.get('right', np.ones(total_frames)), dtype=bool)
        gaze_mask = np.asarray(mask.get('gaze', np.ones(total_frames)), dtype=bool)

    if gaze is not None:
        gaze = np.asarray(gaze, dtype=np.float32)
        if gaze.shape[0] != total_frames:
            raise ValueError(f'gaze frame count mismatch: expected {total_frames}, got {gaze.shape[0]}')

    def _get_hand_frame(hand_data, t):
        if not hand_data:
            return None
        return {name: tf[t] for name, tf in hand_data.items()}

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction('+z')

    def update_frame(t):
        with server.atomic():
            left_frame = _get_hand_frame(left_data, t) if left_mask[t] else None
            right_frame = _get_hand_frame(right_data, t) if right_mask[t] else None
            cam_frame = cam_img[t].copy()
            if gaze is not None and gaze_mask[t] and np.all(np.isfinite(gaze[t])):
                gx, gy = np.round(gaze[t]).astype(np.int32)
                gx, gy = int(gx), int(gy)
                cross_size = 10
                thickness = 3
                cv2.line(cam_frame, (gx - cross_size, gy), (gx + cross_size, gy), GAZE_COLOR, thickness, lineType=cv2.LINE_AA)
                cv2.line(cam_frame, (gx, gy - cross_size), (gx, gy + cross_size), GAZE_COLOR, thickness, lineType=cv2.LINE_AA)

            _render_frame_3d(
                server,
                cam_frame,
                cam_tfs[t],
                left_frame,
                right_frame,
            )

    if total_frames > 1:
        frame_slider = server.gui.add_slider('Frame', min=0, max=total_frames - 1, step=1, initial_value=0)

        @frame_slider.on_update
        def _(_):
            update_frame(int(frame_slider.value))

    update_frame(0)
    print(f'Viser server running at http://localhost:{port}')
    print('Press Ctrl+C to exit')

    while True:
        time.sleep(10.0)


# -----------------------------------------------------------------------------
# 2D keypoint visualization
# -----------------------------------------------------------------------------

def _project_vlia_points_to_pixels(points_vlia, cam_int, eps=1e-6):
    forward = points_vlia[:, 0]
    valid = forward > eps

    uv = np.zeros((points_vlia.shape[0], 2), dtype=np.float32)
    if not np.any(valid):
        return uv, valid

    cam_cv = np.stack(
        [-points_vlia[valid, 1], -points_vlia[valid, 2], points_vlia[valid, 0]],
        axis=1,
    )
    proj = (cam_int @ cam_cv.T).T
    uv[valid, 0] = proj[:, 0] / proj[:, 2]
    uv[valid, 1] = proj[:, 1] / proj[:, 2]
    return uv, valid


def _transform_hand_to_camera_frame(hand_frame_tfs, cam_ext):
    if not hand_frame_tfs:
        return None

    cam_from_world = np.linalg.inv(cam_ext)
    return {
        name: cam_from_world @ tf_world
        for name, tf_world in hand_frame_tfs.items()
    }


def _draw_hand_keypoint_overlay(frame, hand_frame_tfs, color, cam_int):
    if not hand_frame_tfs:
        return

    positions = np.zeros((NUM_HAND_JOINTS, 3), dtype=np.float32)
    joint_valid = np.zeros(NUM_HAND_JOINTS, dtype=bool)

    for name, tf in hand_frame_tfs.items():
        idx = HAND_KEYPOINT_INDEX.get(name)
        if idx is None:
            continue
        positions[idx] = tf[:3, 3]
        joint_valid[idx] = True

    uv, proj_valid = _project_vlia_points_to_pixels(positions, cam_int)
    valid = joint_valid & proj_valid

    for j in range(1, NUM_HAND_JOINTS):
        parent = HAND_PARENT_INDICES[j]
        if parent < 0 or not (valid[parent] and valid[j]):
            continue
        p0 = tuple(np.round(uv[parent]).astype(np.int32))
        p1 = tuple(np.round(uv[j]).astype(np.int32))
        cv2.line(frame, p0, p1, color, 2, lineType=cv2.LINE_AA)

    for j in range(NUM_HAND_JOINTS):
        if not valid[j]:
            continue
        c = tuple(np.round(uv[j]).astype(np.int32))
        cv2.circle(frame, c, 3, color, -1, lineType=cv2.LINE_AA)


def vis_2d_keypoint(cam_img, cam_int, cam_ext, hand, gaze=None, mask=None, save_path=None, fps=30):
    """Overlay hand keypoint skeletons and gaze point on image frames.

    Args:
        cam_img: np.ndarray [T, H, W, 3]
        cam_int: np.ndarray [3, 3]
        cam_ext: np.ndarray [T, 4, 4], camera pose in global/world frame
        hand: {
            'left': {'joint_name': np.ndarray [T, 4, 4], ...} or None,
            'right': {'joint_name': np.ndarray [T, 4, 4], ...} or None,
        }
        gaze: np.ndarray [T,2] image pixel coordinates (optional)
        mask: {'left': np.ndarray [T], 'right': np.ndarray [T], 'gaze': np.ndarray [T]} (optional)
        All hand transforms are in global/world frame and will be converted
        to current-frame camera coordinates before projection.
    """
    cam_int = np.asarray(cam_int, dtype=np.float32)
    cam_ext = np.asarray(cam_ext, dtype=np.float32)

    total_frames = cam_img.shape[0]
    if cam_ext.shape[0] != total_frames:
        raise ValueError(f'cam_ext frame count mismatch: expected {total_frames}, got {cam_ext.shape[0]}')

    try:
        cam_int_frames = np.broadcast_to(cam_int, (total_frames, 3, 3)).astype(np.float32, copy=False)
    except ValueError as exc:
        raise ValueError(f'Expected cam_int shape [3,3], got {cam_int.shape}') from exc

    left_data = hand.get('left') if hand is not None else None
    right_data = hand.get('right') if hand is not None else None

    if mask is None:
        left_mask = np.ones(total_frames, dtype=bool)
        right_mask = np.ones(total_frames, dtype=bool)
        gaze_mask = np.ones(total_frames, dtype=bool)
    else:
        left_mask = np.asarray(mask.get('left', np.ones(total_frames)), dtype=bool)
        right_mask = np.asarray(mask.get('right', np.ones(total_frames)), dtype=bool)
        gaze_mask = np.asarray(mask.get('gaze', np.ones(total_frames)), dtype=bool)

    if gaze is not None:
        gaze = np.asarray(gaze, dtype=np.float32)
        if gaze.shape[0] != total_frames:
            raise ValueError(f'gaze frame count mismatch: expected {total_frames}, got {gaze.shape[0]}')

    def _get_frame_dict(side_data, t):
        if not side_data:
            return None
        return {name: tf_seq[t] for name, tf_seq in side_data.items()}

    rendered = []
    for t in range(total_frames):
        frame = cam_img[t].copy()

        left_world = _get_frame_dict(left_data, t) if left_mask[t] else None
        right_world = _get_frame_dict(right_data, t) if right_mask[t] else None

        left_cam = _transform_hand_to_camera_frame(left_world, cam_ext[t])
        right_cam = _transform_hand_to_camera_frame(right_world, cam_ext[t])

        _draw_hand_keypoint_overlay(frame, left_cam, LEFT_HAND_COLOR, cam_int_frames[t])
        _draw_hand_keypoint_overlay(frame, right_cam, RIGHT_HAND_COLOR, cam_int_frames[t])

        if gaze is not None and gaze_mask[t] and np.all(np.isfinite(gaze[t])):
            gx, gy = np.round(gaze[t]).astype(np.int32)
            gx, gy = int(gx), int(gy)
            cross_size = 6
            thickness = 2
            cv2.line(frame, (gx - cross_size, gy), (gx + cross_size, gy), GAZE_COLOR, thickness, lineType=cv2.LINE_AA)
            cv2.line(frame, (gx, gy - cross_size), (gx, gy + cross_size), GAZE_COLOR, thickness, lineType=cv2.LINE_AA)

        rendered.append(frame)

    rendered = np.asarray(rendered)
    if save_path is not None:
        _save_rgb_video_with_ffmpeg(rendered, save_path=save_path, fps=fps)
    return rendered
