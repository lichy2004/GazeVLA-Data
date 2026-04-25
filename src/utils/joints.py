import numpy as np

# Hand 21 keypoint names: wrist + 5 fingers × 4 joints
HAND_KEYPOINT_NAMES = [
    "wrist",
    "index1", "index2", "index3", "index4",
    "middle1", "middle2", "middle3", "middle4",
    "pinky1", "pinky2", "pinky3", "pinky4",
    "ring1", "ring2", "ring3", "ring4",
    "thumb1", "thumb2", "thumb3", "thumb4",
]

# Parent index for each joint (-1 = root)
HAND_PARENT_INDICES = np.array([
    -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,
])

# Joint name → index lookup
HAND_KEYPOINT_INDEX = {name: i for i, name in enumerate(HAND_KEYPOINT_NAMES)}
