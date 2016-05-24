init: init-apt init-pip

init-apt:
		sudo apt-get install libhdf5-dev libyaml-dev libopencv-dev pkg-config
		sudo apt-get install cmake libsdl1.2-dev
		sudo apt-get install python-dev python-pip libhdf5-dev python-numpy 

init-pip:
		sudo pip install -r requirements.txt
