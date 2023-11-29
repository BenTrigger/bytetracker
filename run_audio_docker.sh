#!/bin/bash
xhost +
sudo docker run \
	--privileged --rm -it  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --network host -e DISPLAY=$DISPLAY --shm-size=48GB \
	--device /dev/video0:/dev/video0 \
	--device /dev/snd/by-id/usb-Plantronics_Poly_Calisto_3200_e366b08cf93c808bab40ca542db7a4bf-01:/dev/snd/by-id/usb-Plantronics_Poly_Calisto_3200_e366b08cf93c808bab40ca542db7a4bf-01 \
	-v /dev/snd/by-id/usb-Plantronics_Poly_Calisto_3200_e366b08cf93c808bab40ca542db7a4bf-01:/dev/snd/by-id/usb-Plantronics_Poly_Calisto_3200_e366b08cf93c808bab40ca542db7a4bf-01 \
       	-v $PWD:/work \
       	-w '/work' \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /tmp/argus_socket:/tmp/argus_socket \
	--name audio_docker_2 \
	--entrypoint /bin/bash \
	ben_docker:latest
