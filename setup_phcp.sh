#!/bin/bash

git clone https://github.com/scibuilder/QD.git
cd QD
./configure CXX=/usr/bin/g++ CXXFLAGS="-fPIC -O3"
make
sudo make install
rm -rf .git
cd ..

sudo apt-get -y install gnat
sed -i '$ a\PATH="/usr/bin/gnat:$PATH"; export PATH' ~/.bashrc
source ~/.bashrc
