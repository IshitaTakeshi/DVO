DATASET_ROOT=datasets
SCRIPT_DIR=tools
SEQUENCE_NAME=rgbd_dataset_freiburg1_desk

wget -c https://vision.in.tum.de/rgbd/dataset/freiburg1/$SEQUENCE_NAME.tgz -P $DATASET_ROOT
tar xzf $DATASET_ROOT/$SEQUENCE_NAME.tgz -C $DATASET_ROOT
python3 $SCRIPT_DIR/associate.py \
    $DATASET_ROOT/$SEQUENCE_NAME/rgb.txt \
    $DATASET_ROOT/$SEQUENCE_NAME/depth.txt \
    >> $DATASET_ROOT/$SEQUENCE_NAME/rgbd.txt
