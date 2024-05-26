sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz

tar -xf Python-3.12.0.tgz
cd Python-3.12.0

./configure --enable-optimizations

make -j $(nproc)
sudo make altinstall

python3.12 --version
