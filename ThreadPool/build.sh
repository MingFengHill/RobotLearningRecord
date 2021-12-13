# shellcheck disable=SC2164
rm -rf bin/*
rm -rf build/*
rm -rf output/*
cd build
cmake .. -DDEBUG=ON
make
make install
