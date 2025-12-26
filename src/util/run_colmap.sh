#!/bin/bash

# 设置工作目录
PROJECT_PATH="/home/hqlab/workspace/reconstruction/result/final_result/parkinglot/scene_0/colmap"
DATABASE="$PROJECT_PATH/database.db"
IMAGES="$PROJECT_PATH/images"
SPARSE="$PROJECT_PATH/sparse"
DENSE="$PROJECT_PATH/dense"
MASK="$PROJECT_PATH/masks"

# 创建必要的目录
mkdir -p "$SPARSE"
mkdir -p "$DENSE"

echo "Step 1: Extracting features..."
colmap feature_extractor \
    --database_path $DATABASE \
    --image_path $IMAGES \
    --ImageReader.mask_path $MASK \
    --ImageReader.camera_model PINHOLE

echo "Step 2: Matching features..."
colmap exhaustive_matcher \
    --database_path $DATABASE 

echo "Step 3: Triangulating points..."
colmap point_triangulator \
    --database_path $DATABASE \
    --image_path $IMAGES \
    --input_path $SPARSE \
    --output_path $SPARSE \
    --Mapper.min_num_matches 5 \
    --Mapper.filter_max_reproj_error 2 \
    --Mapper.filter_min_tri_angle 3

echo "Step 4: Dense reconstruction..."
# Image undistortion
colmap image_undistorter \
    --image_path $IMAGES \
    --input_path $SPARSE \
    --output_path $DENSE \
    --output_type COLMAP

# Stereo matching with reduced depth range
colmap patch_match_stereo \
    --workspace_path $DENSE \
    --PatchMatchStereo.max_image_size 1000 \
    --PatchMatchStereo.window_radius 4 \
    --PatchMatchStereo.window_step 2

# Stereo fusion
colmap stereo_fusion \
    --workspace_path $DENSE \
    --output_path $DENSE/fused.ply \
    --StereoFusion.min_num_pixels 5 \
    --StereoFusion.max_reproj_error 2.0 \
    --StereoFusion.max_depth_error 0.01

echo "COLMAP processing completed!"

# OpenMVS 处理路径设置
OPENMVS_BIN="/home/hqlab/workspace/modeling/openMVS/make/bin"
OPENMVS="$PROJECT_PATH/openmvs"
mkdir -p "$OPENMVS"
SCENE_MVS="scene.mvs"

cd $OPENMVS

echo "Starting OpenMVS processing..."

echo "Step 5: Converting COLMAP output to OpenMVS format..."
"$OPENMVS_BIN/InterfaceCOLMAP" \
    -i "$DENSE" \
    -o "$SCENE_MVS" \
    --image-folder "$DENSE/images"

echo "Step 6: Reconstructing mesh..."
"$OPENMVS_BIN/ReconstructMesh" \
    "$SCENE_MVS" \
    -p "$OPENMVS/scene.ply" 

echo "Step 7: Refining mesh..."
"$OPENMVS_BIN/RefineMesh" \
    "$SCENE_MVS" \
    -m "$OPENMVS/scene_mesh.ply" \
    -o "$OPENMVS/scene_mesh_refine.mvs"

echo "Step 8: Texturing mesh..."
"$OPENMVS_BIN/TextureMesh" \
    "$SCENE_MVS" \
    -m "$OPENMVS/scene_mesh_refine.ply" \
    -o "$OPENMVS/scene_mesh_refine_texture.mvs" \
    --export-type OBJ

echo "OpenMVS processing completed!"
