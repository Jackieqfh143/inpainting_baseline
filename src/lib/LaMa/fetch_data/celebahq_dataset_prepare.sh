mkdir celeba-hq-hq-dataset

unzip data256x256.zip -d celeba-hq-hq-dataset/

# Reindex
for i in `echo {00001..30000}`
do
    mv 'celeba-hq-hq-dataset/data256x256/'$i'.jpg' 'celeba-hq-hq-dataset/data256x256/'$[10#$i - 1]'.jpg'
done


# Split: split train -> train & val
cat fetch_data/train_shuffled.flist | shuf > celeba-hq-hq-dataset/temp_train_shuffled.flist
cat celeba-hq-hq-dataset/temp_train_shuffled.flist | head -n 2000 > celeba-hq-hq-dataset/val_shuffled.flist
cat celeba-hq-hq-dataset/temp_train_shuffled.flist | tail -n +2001 > celeba-hq-hq-dataset/train_shuffled.flist
cat fetch_data/val_shuffled.flist > celeba-hq-hq-dataset/visual_test_shuffled.flist

mkdir celeba-hq-hq-dataset/train_256/
mkdir celeba-hq-hq-dataset/val_source_256/
mkdir celeba-hq-hq-dataset/visual_test_source_256/

cat celeba-hq-hq-dataset/train_shuffled.flist | xargs -I {} mv celeba-hq-hq-dataset/data256x256/{} celeba-hq-hq-dataset/train_256/
cat celeba-hq-hq-dataset/val_shuffled.flist | xargs -I {} mv celeba-hq-hq-dataset/data256x256/{} celeba-hq-hq-dataset/val_source_256/
cat celeba-hq-hq-dataset/visual_test_shuffled.flist | xargs -I {} mv celeba-hq-hq-dataset/data256x256/{} celeba-hq-hq-dataset/visual_test_source_256/


# create location config celeba-hq.yaml
PWD=$(pwd)
DATASET=${PWD}/celeba-hq-hq-dataset
CELEBA=${PWD}/configs/training/location/celeba-hq.yaml

touch $CELEBA
echo "# @package _group_" >> $CELEBA
echo "data_root_dir: ${DATASET}/" >> $CELEBA
echo "out_root_dir: ${PWD}/experiments/" >> $CELEBA
echo "tb_dir: ${PWD}/tb_logs/" >> $CELEBA
echo "pretrained_models: ${PWD}/" >> $CELEBA
