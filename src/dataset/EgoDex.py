import fnmatch
import os
import sys

import cv2
import h5py
import numpy as np
from tqdm import tqdm

sys.path.append(".")

from src.utils.data_format import get_vlia_data_template, image_encoding, save_vlia_hdf5
from src.utils.transform import create_rotation_transform
from src.utils.joints import HAND_KEYPOINT_NAMES


JOINT_MAPPING = {
    "wrist": "Hand",
    "index1": "IndexFingerKnuckle",
    "index2": "IndexFingerIntermediateBase",
    "index3": "IndexFingerIntermediateTip",
    "index4": "IndexFingerTip",
    "middle1": "MiddleFingerKnuckle",
    "middle2": "MiddleFingerIntermediateBase",
    "middle3": "MiddleFingerIntermediateTip",
    "middle4": "MiddleFingerTip",
    "pinky1": "LittleFingerKnuckle",
    "pinky2": "LittleFingerIntermediateBase",
    "pinky3": "LittleFingerIntermediateTip",
    "pinky4": "LittleFingerTip",
    "ring1": "RingFingerKnuckle",
    "ring2": "RingFingerIntermediateBase",
    "ring3": "RingFingerIntermediateTip",
    "ring4": "RingFingerTip",
    "thumb1": "ThumbKnuckle",
    "thumb2": "ThumbIntermediateBase",
    "thumb3": "ThumbIntermediateTip",
    "thumb4": "ThumbTip",
}


def process_text(data, root):
    if root.attrs.get("llm_type") == "reversible":
        direction = root.attrs.get("which_llm_description", "1")
        lang_instruct = root.attrs.get(
            "llm_description" if direction == "1" else "llm_description2", ""
        )
    else:
        lang_instruct = root.attrs.get("llm_description", "")

    if isinstance(lang_instruct, bytes):
        lang_instruct = lang_instruct.decode("utf-8")

    data["text"] = lang_instruct


def process_camera(data, root):
    data["camera"]["intrinsic"] = root["/camera/intrinsic"][:]
    data["camera"]["extrinsic"] = root["/transforms/camera"][:]


def process_hand_keypoint(data, root):
    num_frames = data["metadata"]["frames"]
    left_exist = "/transforms/leftHand" in root
    right_exist = "/transforms/rightHand" in root
    data["mask"]["hand"] = np.array(
        [[int(left_exist), int(right_exist)] for _ in range(num_frames)], dtype=np.int32
    )

    joint_tfs_world = {}
    for hand_name in HAND_KEYPOINT_NAMES:
        joint_tfs_world[f"left_{hand_name}"] = root[
            f"/transforms/left{JOINT_MAPPING[hand_name]}"
        ][:]
        joint_tfs_world[f"right_{hand_name}"] = root[
            f"/transforms/right{JOINT_MAPPING[hand_name]}"
        ][:]

    joint_tfs_world["camera"] = root["/transforms/camera"][:]

    # global coordinate rotation
    camera_tfs_first = joint_tfs_world["camera"][0:1]
    camera_tfs_first = (
        camera_tfs_first
        @ create_rotation_transform("x", -90)
        @ create_rotation_transform("z", 90)
    )

    # first frame align
    joint_tfs_camera = {}
    camera_tfs_first_inv = np.linalg.inv(camera_tfs_first)
    for joint_name, joint_tf in joint_tfs_world.items():
        joint_tfs_camera[joint_name] = camera_tfs_first_inv @ joint_tf

    # local coordinate rotation
    joint_tfs_camera["camera"] = (
        joint_tfs_camera["camera"]
        @ create_rotation_transform("x", -90)
        @ create_rotation_transform("z", 90)
    )
    joint_tfs_camera["left_wrist"] = (
        joint_tfs_camera["left_wrist"] 
        @ create_rotation_transform("x", -90)
    )
    joint_tfs_camera["right_wrist"] = (
        joint_tfs_camera["right_wrist"]
        @ create_rotation_transform("y", 180)
        @ create_rotation_transform("x", 90)
    )

    # save hand keypoint
    left_hand_tfs = {}
    right_hand_tfs = {}
    for hand_name in HAND_KEYPOINT_NAMES:
        left_hand_tfs[hand_name] = joint_tfs_camera[f"left_{hand_name}"]
        right_hand_tfs[hand_name] = joint_tfs_camera[f"right_{hand_name}"]

    data["hand"]["keypoint"]["left"] = left_hand_tfs
    data["hand"]["keypoint"]["right"] = right_hand_tfs
    data["camera"]["extrinsic"] = joint_tfs_camera["camera"]


def process_gaze(data, root):
    num_frames = data["metadata"]["frames"]
    data["gaze"] = np.zeros((num_frames, 2), dtype=np.float32)
    data["mask"]["gaze"] = np.zeros((num_frames, 1), dtype=np.int32)


def process_images(data, mp4_path):
    cap = cv2.VideoCapture(mp4_path)

    encode_data = []
    max_len = 0
    num_frames = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        jpeg_data = image_encoding(frame_bgr)
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
        num_frames += 1

    cap.release()

    padded_data = [d.ljust(max_len, b"\0") for d in encode_data]
    data["images"] = padded_data
    return max_len, num_frames


def process_metadata(data, frame_count, hdf5_path):
    data["metadata"]["frames"] = frame_count
    data["metadata"]["data_path"] = hdf5_path


def process_episode(hdf5_path, output_path):
    mp4_path = hdf5_path.replace(".hdf5", ".mp4")
    data = get_vlia_data_template()

    len_high, frame_count = process_images(data, mp4_path)
    process_metadata(data, frame_count, hdf5_path)

    with h5py.File(hdf5_path, "r") as root:
        process_text(data, root)
        process_camera(data, root)
        process_hand_keypoint(data, root)
        process_gaze(data, root)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_vlia_hdf5(output_path, data, len_high=len_high)


def collect_files(src_dir):
    hdf5_files = []
    for root, _, files in os.walk(src_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(os.path.join(root, filename))

    print(f"Found {len(hdf5_files)} HDF5 files to process.")
    hdf5_files.sort()
    return hdf5_files


def main(raw_dir, out_dir):
    hdf5_files = collect_files(raw_dir)
    for index, hdf5_path in tqdm(enumerate(hdf5_files), desc="Processing EgoDex Dataset"):
        dst_path = os.path.join(out_dir, f"{index:08d}.hdf5")
        process_episode(hdf5_path, dst_path)


if __name__ == "__main__":
    src_dir = "data/EgoDex/raw"
    dst_dir = "data/EgoDex/processed"
    main(src_dir, dst_dir)
