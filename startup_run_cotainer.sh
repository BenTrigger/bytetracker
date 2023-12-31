#!/bin/bash
cd /home/user1/ariel/byte_tracker_for_einat_0.3/
xhost +
sudo docker run \
	--privileged --rm -it  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --network host -e DISPLAY=$DISPLAY --shm-size=48GB \
	--device /dev/video0 \
	--device /dev/snd/by-id/usb-Plantronics_Poly_Calisto_3200_0d3d6b4505e04f91874f47c233f6a3a3-0:/dev/snd/by-id/usb-Plantronics_Poly_Calisto_3200_0d3d6b4505e04f91874f47c233f6a3a3-0 \
	-v /dev/snd/by-id/usb-Plantronics_Poly_Calisto_3200_0d3d6b4505e04f91874f47c233f6a3a3-0:/dev/snd/by-id/usb-Plantronics_Poly_Calisto_3200_0d3d6b4505e04f91874f47c233f6a3a3-0 \
       	-v $PWD:/work \
       	-w '/work' \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /tmp/argus_socket:/tmp/argus_socket \
	--name audio_docker_1 \
	--entrypoint /bin/bash \
	ben_docker:latest \
	./start.sh
