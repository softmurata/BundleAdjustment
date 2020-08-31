import os
import subprocess
import numpy as np
from colmap_read_model import read_cameras_binary, read_images_binary, read_points3d_binary

# scene directory

# extract featrues
# match features
# gen pose



def run_colmap(basedir):
    """
    $ DATASET_PATH=/path/to/dataset
    $ colmap feature_extractor \
        --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images
    $ colmap exhaustive_matcher \
        --database_path $DATASET_PATH/database.db
    $ mkdir $DATASET_PATH/sparse
    $ colmap mapper \
        --database_path $DATASET_PATH/database.db \
        --image_path $DATASET_PATH/images \
        --output_path $DATASET_PATH/sparse
    """
    # prepare the log file
    log_file_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(log_file_name, 'w')
    
    feature_extractor_args = ['colmap', 'feature_extractor', '--database_path', os.path.join(basedir, 'database.db'),
                              '--image_path', os.path.join(basedir, images), '--ImageReder.single_camera', '1']
    
    feat_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True))
    logfile.write(feat_output)
    print('Feature extraction process was finished')
    
    exhaustive_matcher_args = ['colmap', 'exhaustive_matcher', '--database_path', os.path.join(basedir, 'database.db')]
    
    match_output = (subprocess.check_output(exhaustive_matcher_args, universal_newlines=True))
    logfile.write(match_output)
    print('Exhaustive match was finished')
    
    sparse_dir = os.path.join(basedir, 'sparse')
    
    os.makedirs(sparse_dir, exist_ok=True)
    
    mapper_args = ['colmap', 'mapper', '--database_path', os.path.join(basedir, 'database.db'),
                   '--image_path', os.path.join(basedir, 'images'), '--output_path', os.path.join(basedir, 'sparse'), 
                   '--Mapper.num_threads', '16',
                   '--Mapper.init_min_tri_angle', '4',
                   '--Mapper.multiple_models', '0',
                   '--Mapper.extract_colors', '0']
    map_output = (subprocess.check_output(mapper_args, universal_newlines=True))
    logfile.write(map_output)
    
    print('sparse map created')
    
    logfile.close()
    
    print()
    print('---finish colmap process----')
    


def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_cameras_binary(camerasfile)
    
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print('cameras:', len(cam))
    
    h, w, f = cam.height, cam.width, cam.params[0]  # f is focal length
    hwf = np.array([h, w, f]).reshape((3, 1))
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imgdata = read_images_binary(imagesfile)
    
    w2c_mats = []  # transformation matrix for converting world to camera
    bottom = np.array([0, 0, 0, 1]).reshape((1, 4))
    names = [imgdata[k].name for k in imgdata]
    
    print('Images #', len(names))
    
    perm = np.argsort(names)
    
    for k in imgdata:
        img = imgdata[k]
        R = img.qvec2rotmat()
        t = img.tvec.reshape((3, 1))
        # create 4 * 4 matrix
        m = np.concatenate([R, t], 1)
        m = np.concatenate([m, bottom], 0)
        w2c_mats.append(m)
        
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose((1, 2, 0))
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_points3d_binary(points3dfile)
    
    # have to switch to [-u, r, -t] from [r, -u, t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm

def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = [] # camera matrix(?)
    
    for k in pts3d:
        # add 3d point coordinate
        pts_arr.append(pts3d[k].xyz)
        
        cams = [0] * poses.shape[1]
        
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        
        vis_arr.append(cams)
        
    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    
    print('Points:', pts_arr.shape, ' Visibility:', vis_arr.shape)
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose((2, 0, 1)) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    
    print('Depth stats:', valid_z.min(), valid_z.max(), valid_z.mean())
    
    # save
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, 0.1), np.percentile(zs, 99.9)
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
        
    save_arr = np.array(save_arr)
    
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)

def minify():
    pass

def main():
    basedir = ''
    factors = None

    # Initialize required file(get by colmap)
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]

    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
        
    # create colmap data(poses, images, points3d)
    if not all([f in files_had for f in files_needed]):
        print('need to run colmap')
        run_colmap(basedir)
        
    else:
        print('do not need run colmap')
        
    poses, pts3d, perm = load_colmap_data(basedir)

    save_poses(basedir, poses, pts3d, perm)

    if factors is not None:
        print('Factors:', factors)
        minify(basedir, factors)
        
    print('finish img2pose')
    
    
if __name__ == '__main__':
    main()
