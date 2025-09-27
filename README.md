# GOLexp_pytorch

필요한 것:
cmake, make, python3.12 

설정 스크립트:

python3 -m venv myexp
source ./myexp/bin/activate
pip3 install torch pygame

mkdir build
cd build
cmake ..
make -j$(nproc)

