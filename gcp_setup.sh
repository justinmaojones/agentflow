sudo apt-get install -y tmux python-dev python3-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py --user
python3 get-pip.py --user
echo "
export PATH=$PATH:/home/$USER/.local/bin
" >> ~/.bashrc
source ~/.bashrc
pip3 install virtualenv --user
pip3 install virtualenvwrapper --user
echo "
export VIRTUALENVWRAPPER_PYTHON=$(which python3)
source /home/$USER/.local/bin/virtualenvwrapper.sh
" >> ~/.bashrc
source ~/.bashrc
