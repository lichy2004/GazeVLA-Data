import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(".")
sys.path.append("src/utils/hot3d")

from projectaria_tools.core.calibration import LINEAR
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from data_loaders.AriaDataProvider import AriaDataProvider
from data_loaders.HeadsetPose3dProvider import load_headset_pose_provider_from_csv
from data_loaders.ManoHandDataProvider import MANOHandDataProvider
from data_loaders.mano_layer import MANOHandModel

from src.utils.data_format import get_vlia_data_template, image_encoding, save_vlia_hdf5
from src.utils.joints import HAND_KEYPOINT_NAMES
from src.utils.transform import create_rotation_transform

MANO_MODEL_DIR = "assets/mano_v1_2/models"

HOT3D_MANO_TO_VLIA = {
    "wrist": 5,
    "index1": 8,
    "index2": 9,
    "index3": 10,
    "index4": 1,
    "middle1": 11,
    "middle2": 12,
    "middle3": 13,
    "middle4": 2,
    "pinky1": 14,
    "pinky2": 15,
    "pinky3": 16,
    "pinky4": 3,
    "ring1": 17,
    "ring2": 18,
    "ring3": 19,
    "ring4": 4,
    # "thumb1": 1,
    "thumb2": 6,
    "thumb3": 7,
    "thumb4": 0,
}

def _load_sequence_metadata(sequence_dir):
    metadata_path = os.path.join(sequence_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_rgb_stream_id(aria_provider):
    stream_ids = aria_provider.get_image_stream_ids()
    for stream_id in stream_ids:
        if aria_provider.get_image_stream_label(stream_id) == "camera-rgb":
            return stream_id

    if len(stream_ids) == 0:
        raise RuntimeError("No camera stream found in HOT3D recording.vrs")

    return stream_ids[0]


def process_text(data, metadata):
    object_names = metadata.get("object_names", [])
    if isinstance(object_names, list) and len(object_names) > 0:
        data["text"] = "Interact with objects: " + ", ".join(object_names)
    else:
        data["text"] = ""


def process_images(data, aria_provider, stream_id):
    timestamps = aria_provider.get_sequence_timestamps(
        stream_id=stream_id,
        time_domain=TimeDomain.TIME_CODE,
    )

    encode_data = []
    max_len = 0
    image_shape = None

    for timestamp_ns in timestamps:
        frame_rgb = aria_provider.get_image(timestamp_ns, stream_id)
        if image_shape is None:
            image_shape = frame_rgb.shape[:2]

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

        jpeg_data = image_encoding(frame_bgr)
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))

    padded_data = [d.ljust(max_len, b"\0") for d in encode_data]
    data["images"] = padded_data

    return timestamps, max_len, image_shape


def process_camera(data, aria_provider, headset_pose_provider, stream_id, timestamps):
    _, linear_calib = aria_provider.get_camera_calibration(stream_id, camera_model=LINEAR)
    focal_lengths = linear_calib.get_focal_lengths()
    principal_point = linear_calib.get_principal_point()

    intrinsic = np.array(
        [
            [focal_lengths[0], 0.0, principal_point[0]],
            [0.0, focal_lengths[1], principal_point[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    T_device_camera, _ = aria_provider.get_camera_calibration(stream_id)
    T_device_camera = T_device_camera.to_matrix().astype(np.float32)

    extrinsic = np.tile(np.eye(4, dtype=np.float32), (len(timestamps), 1, 1))
    for i, timestamp_ns in enumerate(timestamps):
        pose_with_dt = headset_pose_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
            acceptable_time_delta=2_000_000,
        )
        if pose_with_dt is None:
            continue

        T_world_device = pose_with_dt.pose3d.T_world_device.to_matrix().astype(np.float32)
        extrinsic[i] = T_world_device @ T_device_camera

    data["camera"]["intrinsic"] = intrinsic
    data["camera"]["extrinsic"] = extrinsic


def _init_joint_tfs(num_frames):
    joint_tfs = {}
    for hand_side in ["left", "right"]:
        for joint_name in HAND_KEYPOINT_NAMES:
            joint_tfs[f"{hand_side}_{joint_name}"] = np.tile(
                np.eye(4, dtype=np.float32),
                (num_frames, 1, 1),
            )
    return joint_tfs


def _get_mano_hand_model(mano_model_dir):
    mano_left = os.path.join(mano_model_dir, "MANO_LEFT.pkl")
    mano_right = os.path.join(mano_model_dir, "MANO_RIGHT.pkl")
    if os.path.exists(mano_left) and os.path.exists(mano_right):
        return MANOHandModel(mano_model_dir)
    raise FileNotFoundError(
        f"Cannot find MANO_LEFT.pkl / MANO_RIGHT.pkl in {mano_model_dir}"
    )


def process_hand_keypoint(data, mano_provider, timestamps):
    num_frames = data["metadata"]["frames"]
    joint_tfs_world = _init_joint_tfs(num_frames)
    joint_tfs_world["camera"] = data["camera"]["extrinsic"].copy()

    hand_mask = np.zeros((num_frames, 2), dtype=np.int32)

    for frame_idx, timestamp_ns in enumerate(timestamps):
        hand_pose_with_dt = mano_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
            acceptable_time_delta=2_000_000,
        )
        if hand_pose_with_dt is None:
            continue

        hand_pose_collection = hand_pose_with_dt.pose3d_collection

        for handedness, hand_pose in hand_pose_collection.poses.items():
            hand_landmarks = mano_provider.get_hand_landmarks(hand_pose)
            if hand_landmarks is None:
                continue

            hand_landmarks = hand_landmarks.detach().cpu().numpy().astype(np.float32)
            hand_side = "left" if int(handedness.value) == 0 else "right"
            hand_mask[frame_idx, int(handedness.value)] = 1

            for joint_name in HAND_KEYPOINT_NAMES:
                if joint_name == "thumb1":
                    continue
                lm_index = HOT3D_MANO_TO_VLIA[joint_name]
                joint_tfs_world[f"{hand_side}_{joint_name}"][frame_idx, :3, 3] = hand_landmarks[
                    lm_index
                ]

            joint_tfs_world[f"{hand_side}_thumb1"][frame_idx, :3, 3] = 0.5 * (
                joint_tfs_world[f"{hand_side}_wrist"][frame_idx, :3, 3] + joint_tfs_world[f"{hand_side}_thumb2"][frame_idx, :3, 3]
            )

    data["mask"]["hand"] = hand_mask

    camera_tfs_first = joint_tfs_world["camera"][0:1]
    camera_tfs_first = (
        camera_tfs_first
        @ create_rotation_transform("y", 90)
    )

    joint_tfs_camera = {}
    camera_tfs_first_inv = np.linalg.inv(camera_tfs_first)
    for joint_name, joint_tf in joint_tfs_world.items():
        joint_tfs_camera[joint_name] = camera_tfs_first_inv @ joint_tf

    joint_tfs_camera["camera"] = (
        joint_tfs_camera["camera"]
        @ create_rotation_transform("y", 90)
    )
    joint_tfs_camera["left_wrist"] = (
        joint_tfs_camera["left_wrist"]
        @ create_rotation_transform("z", 180)
    )
    joint_tfs_camera["right_wrist"] = (
        joint_tfs_camera["right_wrist"]
        @ create_rotation_transform("z", 180)
    )

    left_hand_tfs = {}
    right_hand_tfs = {}
    for hand_name in HAND_KEYPOINT_NAMES:
        left_hand_tfs[hand_name] = joint_tfs_camera[f"left_{hand_name}"]
        right_hand_tfs[hand_name] = joint_tfs_camera[f"right_{hand_name}"]

    data["hand"]["keypoint"]["left"] = left_hand_tfs
    data["hand"]["keypoint"]["right"] = right_hand_tfs
    data["camera"]["extrinsic"] = joint_tfs_camera["camera"]


def process_gaze(data, aria_provider, stream_id, timestamps, image_shape):
    gaze = np.zeros((len(timestamps), 2), dtype=np.float32)
    gaze_mask = np.zeros((len(timestamps), 1), dtype=np.int32)

    image_height, image_width = image_shape

    for i, timestamp_ns in enumerate(timestamps):
        gaze_xy = aria_provider.get_eye_gaze_in_camera(stream_id, timestamp_ns)
        if gaze_xy is None:
            continue

        x_raw = float(gaze_xy[0])
        y_raw = float(gaze_xy[1])

        if not (0.0 <= x_raw < image_width and 0.0 <= y_raw < image_height):
            continue

        x_rot = image_height - y_raw
        y_rot = x_raw

        gaze[i, 0] = y_rot
        gaze[i, 1] = x_rot
        gaze_mask[i, 0] = 1

    data["gaze"] = gaze
    data["mask"]["gaze"] = gaze_mask


def process_metadata(data, frame_count, sequence_dir):
    data["metadata"]["frames"] = frame_count
    data["metadata"]["data_path"] = sequence_dir


def process_episode(sequence_dir, output_path):
    metadata = _load_sequence_metadata(sequence_dir)

    aria_provider = AriaDataProvider(
        vrs_filepath=os.path.join(sequence_dir, "recording.vrs"),
        mps_folder_path=os.path.join(sequence_dir, "mps"),
    )
    headset_pose_provider = load_headset_pose_provider_from_csv(
        os.path.join(sequence_dir, "headset_trajectory.csv")
    )
    mano_provider = MANOHandDataProvider(
        os.path.join(sequence_dir, "mano_hand_pose_trajectory.jsonl"),
        _get_mano_hand_model(MANO_MODEL_DIR),
    )

    stream_id = _get_rgb_stream_id(aria_provider)

    data = get_vlia_data_template()

    timestamps, len_high, image_shape = process_images(data, aria_provider, stream_id)
    process_metadata(data, len(timestamps), sequence_dir)
    process_text(data, metadata)
    process_camera(data, aria_provider, headset_pose_provider, stream_id, timestamps)
    process_hand_keypoint(data, mano_provider, timestamps)
    process_gaze(data, aria_provider, stream_id, timestamps, image_shape)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_vlia_hdf5(output_path, data, len_high=len_high)


def collect_files(src_dir):
    sequence_dirs = []
    for root, _, files in os.walk(src_dir):
        if "metadata.json" not in files:
            continue

        required_names = {
            "recording.vrs",
            "headset_trajectory.csv",
            "mano_hand_pose_trajectory.jsonl",
        }
        if all(name in files for name in required_names):
            sequence_dirs.append(root)

    print(f"Found {len(sequence_dirs)} HOT3D sequences to process.")
    sequence_dirs.sort()
    return sequence_dirs


def main(raw_dir, out_dir):
    sequence_dirs = collect_files(raw_dir)
    for index, sequence_dir in tqdm(
        enumerate(sequence_dirs),
        desc="Processing HOT3D Dataset",
    ):
        dst_path = os.path.join(out_dir, f"{index:08d}.hdf5")
        process_episode(sequence_dir, dst_path)


if __name__ == "__main__":
    src_dir = "data/HOT3D/raw"
    dst_dir = "data/HOT3D/processed"
    main(src_dir, dst_dir)
