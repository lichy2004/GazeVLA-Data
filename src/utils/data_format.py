import cv2
import h5py
import numpy as np


def image_encoding(img) -> bytes:
    """
    Encode a single image (H, W, C) BGR numpy array to JPEG bytes.

    Args:
        img: numpy array (H, W, C) in BGR format, or a torch Tensor.

    Returns:
        bytes: raw JPEG bytes.
    """
    # Convert torch tensor to numpy if needed
    if hasattr(img, 'numpy'):
        img = img.numpy()

    success, encoded_image = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Failed to encode image")

    return encoded_image.tobytes()


def image_decoding(img_bytes: bytes, to_rgb: bool = True) -> np.ndarray:
    """
    Decode a single JPEG bytes (possibly zero-padded) back to a numpy image.

    Args:
        img_bytes: raw or zero-padded JPEG bytes.
        to_rgb:    if True, convert BGR -> RGB before returning.

    Returns:
        numpy array (H, W, C) in RGB (or BGR if to_rgb=False) format.
        Returns a black image (1080x1920x3) if decoding fails.
    """
    img_bytes = img_bytes.rstrip(b'\x00')
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        # Fallback to a black placeholder
        placeholder = np.zeros((1080, 1920, 3), dtype=np.uint8)
        return placeholder

    if to_rgb:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr


def images_encoding(imgs):
    """
    Encode a sequence of images to JPEG bytes with zero padding.

    Args:
        imgs: list/array of (H, W, C) BGR numpy arrays or torch Tensors.

    Returns:
        padded_data: list of zero-padded byte strings (all same length).
        max_len:     int, the length of the longest encoded image in bytes.
    """
    encode_data = [image_encoding(img) for img in imgs]
    max_len = max(len(d) for d in encode_data)
    padded_data = [d.ljust(max_len, b"\0") for d in encode_data]
    return padded_data, max_len

def get_vlia_data_template():
    return {
        'images': None,     # List of bytes, encoded images
        'text': "",         # str
        'camera': {
            'intrinsic': None,  # [3, 3]
            'extrinsic': None,  # [N, 4, 4]
        },
        'hand': {
            'keypoint': {
                'left': None,   # dict
                'right': None,  # dict
            }
        },
        'gaze': None,       # [N, 2]
        'mask': {
            'hand': None,   # [N, 2] -> index 0=left, 1=right
            'gaze': None,   # [N, 1]
        },
        'metadata': {
            'data_path': "",  # str
            'frames': 0,      # int
        }
    }

def save_vlia_hdf5(file_path, data, len_high=None):
    with h5py.File(file_path, 'w') as f:
        # Images
        if data['images'] is not None and len_high is not None:
            f.create_dataset('images', data=data['images'], dtype=f"S{len_high}")
        
        # Text
        f.attrs['text'] = data['text']
        
        # Camera
        cam_grp = f.create_group('camera')
        if data['camera']['intrinsic'] is not None:
            cam_grp.create_dataset('intrinsic', data=data['camera']['intrinsic'])
        if data['camera']['extrinsic'] is not None:
            cam_grp.create_dataset('extrinsic', data=data['camera']['extrinsic'])
            
        # Hand
        hand_grp = f.create_group('hand')
        keypoint_grp = hand_grp.create_group('keypoint')
        for side in ['left', 'right']:
            side_data = data['hand']['keypoint'][side]
            if side_data is None:
                continue
            side_grp = keypoint_grp.create_group(side)
            for k, v in side_data.items():
                side_grp.create_dataset(k, data=v)
                
        # Gaze
        if data['gaze'] is not None:
            f.create_dataset('gaze', data=data['gaze'])
            
        # Mask
        mask_grp = f.create_group('mask')
        if data['mask']['hand'] is not None:
            mask_grp.create_dataset('hand', data=data['mask']['hand'])
        if data['mask']['gaze'] is not None:
            mask_grp.create_dataset('gaze', data=data['mask']['gaze'])
            
        # Metadata
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['data_path'] = data['metadata']['data_path']
        meta_grp.attrs['frames'] = data['metadata']['frames']