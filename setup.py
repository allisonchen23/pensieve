import os

start_dir = os.getcwd()

# mahimahi
os.system("sysctl -w net.ipv4.ip_forward=1")
os.system("add-apt-repository -y ppa:keithw/mahimahi")
# os.system("apt-get -y update")
# os.system("apt-get -y install mahimahi")
print("command: yum update")
os.system("yum update")
print("command: yum install mahimahi")
os.system("yum install mahimahi")

# apache server
# os.system("apt-get -y install apache2")
print("command: yum install apache2")
os.system("yum install apache2")

# selenium
os.system("wget 'https://pypi.python.org/packages/source/s/selenium/selenium-2.39.0.tar.gz'")
# os.system("apt-get -y install python-setuptools python-pip xvfb xserver-xephyr tightvncserver unzip")
print("command: yum install python-setuptools python-pip xvfb xserver-xephyr tightvncserver unzip")
os.system("yum install python-setuptools python-pip xvfb xserver-xephyr tightvncserver unzip")

os.system("tar xvzf selenium-2.39.0.tar.gz")
selenium_dir = start_dir + "/selenium-2.39.0"
os.chdir( selenium_dir )
os.system("python setup.py install" )
os.system("sh -c \"echo 'DBUS_SESSION_BUS_ADDRESS=/dev/null' > /etc/init.d/selenium\"")

# py virtual display
os.chdir( start_dir )
os.system("pip install pyvirtualdisplay")
os.system("wget 'https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb' ")
os.system("dpkg -i google-chrome-stable_current_amd64.deb")
# os.system("apt-get -f -y install")
os.system("yum install")

# tensorflow
os.system("apt-get -y install python-pip python-dev")
os.system("yum install python-pip python-dev")
os.system("pip install tensorflow==1.1.0")

# tflearn
os.system("pip install tflearn==0.3.1")
# os.system("apt-get -y install python-h5py")
# os.system("apt-get -y install python-scipy")
os.system("yum install python-h5py")
os.system("yum install python-scipy")

# matplotlib
# os.system("apt-get -y install python-matplotlib")
os.system("yum install python-matplotlib")

# copy the webpage files to var/www/html
os.chdir( start_dir )
os.system("cp video_server/myindex_*.html var/www/html")
os.system("cp video_server/dash.all.min.js var/www/html")
os.system("cp -r video_server/video* var/www/html")
os.system("cp video_server/Manifest.mpd var/www/html")

# make results directory
os.system("mkdir cooked_traces")
os.system("mkdir rl_server/results")
os.system("mkdir run_exp/results")
os.system("mkdir real_exp/results")

# need to copy the trace and pre-trained NN model
print("Need to put trace files in 'pensieve/cooked_traces'.")
print("Need to put pre-trained NN model in 'pensieve/rl_server/results'.")
