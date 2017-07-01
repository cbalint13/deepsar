#!/usr/bin/bash


export PYTHONPATH=${PYTHONPATH}:`pwd`

./make_unet.py

sed -e 's|EuclideanLoss1|TrainLoss|g' -i unet.prototxt
sed -e 's|EuclideanLoss1|TestLoss|g' -i unet_tests.prototxt

sed -e 's|Python1|radar|g' -i unet.prototxt
sed -e 's|Python2|image|g' -i unet.prototxt
sed -e 's|HDF5Data1|radar|g' -i unet.prototxt
sed -e 's|HDF5Data2|image|g' -i unet.prototxt
sed -e 's|HDF5Data1|radar|g' -i unet_tests.prototxt
sed -e 's|HDF5Data2|image|g' -i unet_tests.prototxt
sed -e 's|Input1|radar|g' -i unet_infer.prototxt

ls hdf5-test/* > test.txt

snap=`ls -lAtr snapshots/* | tail -1 | awk '{print $9}'`

if [ ${#snap} -gt 5 ]
then
 echo "Train resume from: [$snap]"
 caffe train -solver=unet_solver.prototxt -snapshot $snap  -sighup_effect none -sigint_effect stop 2>&1 | tee -a caffe.log
else
 echo "Train from begin"
 caffe train -solver=unet_solver.prototxt 2>&1 | tee caffe.log
fi
