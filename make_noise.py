import subprocess

def make_noise():
    cmd = "gst-launch-1.0 filesrc location=Drone_record.wav ! wavparse ! audioconvert ! audioresample ! alsasink device=hw:0"
    x =subprocess.Popen(cmd, shell=True)
make_noise()
