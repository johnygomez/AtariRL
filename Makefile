init: init-apt init-pip

init-apt:
		sudo apt-get install python-pip libhdf5-dev python-numpy 

init-pip:
		sudo pip install -r requirements.txt
