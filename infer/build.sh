BUILD_TYPE=${1:-Release}

echo "+++++++++++++++++++++++++++++";
echo ${BUILD_TYPE}

mkdir build || True
cd build
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..
make -j 4