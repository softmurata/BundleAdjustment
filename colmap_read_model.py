from collections import namedtuple
import struct
import numpy as np


# model
CameraModel = namedtuple("CameraModel", ['model_id', 'model_name', 'num_params'])

Camera = namedtuple("Camera", ['id', 'model', 'width', 'height', 'params'])

BaseImage = namedtuple("Image", ['id', 'qvec', 'tvec', 'camera_id', 'name', 'xys', 'point3D_ids'])

Point3D = namedtuple("Point3D", ['id', 'xyz', 'rgb', 'error', 'image_ids', 'point2D_idxs'])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    
    def qvec2rotmat(self):
        
        return qvec2rotmat(self.qvec)
    
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name='SIMPLE_PINHOLE', num_params=3),
    CameraModel(model_id=1, model_name='PINHOLE', num_params=4),
    CameraModel(model_id=2, model_name='SIMPLE_RADIAL', num_params=4),
    CameraModel(model_id=3, model_name='RADIAL', num_params=5),
    CameraModel(model_id=4, model_name='OPENCV', num_params=8),
    CameraModel(model_id=5, model_name='OPENCV_FISHEYE', num_params=8),
    CameraModel(model_id=6, model_name='FULL_OPENCV', num_params=12),
    CameraModel(model_id=7, model_name='FOV', num_params=5),
    CameraModel(model_id=8, model_name='SIMPLE_RADIAL_FISHEYE', num_params=4),
    CameraModel(model_id=9, model_name='RADIAL_FISHEYE', num_params=5),
    CameraModel(model_id=10, model_name='THIN_PRISM_FISHEYE', num_params=12)
}

CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])



def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character='<'):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(camerasfile):
    camdata = {}
    
    with open(camerasfile, 'rb') as fid:
        num_cameras = read_next_bytes(fid, 8, 'Q')[0]
        
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence='iiQQ')
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence='d'*num_params)
            camdata[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))
            
    
    return camdata

def read_images_binary(imagesfile):
    imgdata = {}
    
    with open(imagesfile, 'rb') as fid:
        num_reg_images = read_next_bytes(fid, 8, 'Q')[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence='idddddddi')
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ''
            current_char = read_next_bytes(fid, 1, 'c')[0]
            
            while current_char != b'\x00':
                image_name += current_char.decode('utf-8')
                current_char = read_next_bytes(fid, 1, 'c')[0]
                
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence='ddq' * num_points2D)
            
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            imgdata[image_id] = Image(id=image_id,
                                      qvec=qvec,
                                      tvec=tvec,
                                      camera_id=camera_id,
                                      name=image_name,
                                      xys=xys,
                                      point3D_ids=point3D_ids)
    
    return imgdata

def read_points3d_binary(points3dfile):
    
    pts3d = {}
    
    with open(points3dfile, 'rb') as fid:
        num_points = read_next_bytes(fid, 8, 'Q')[0]
        
        for point_line_index in range(num_points):
            
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence='QdddBBBd')
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            track_elems = read_next_bytes(fid, num_bytes=8*track_length, format_char_sequence='ii' * track_length)
            
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            
            
            pts3d[point3D_id] = Point3D(id=point3D_id,
                                        xyz=xyz,
                                        rgb=rgb,
                                        error=error,
                                        image_ids=image_ids,
                                        point2D_idxs=point2D_idxs)
    
    
    return pts3d

