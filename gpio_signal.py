import Jetson.GPIO as GPIO
import time

time_to_check = 5
# Set the GPIO mode to BCM



# Define the GPIO pin number
#not_working = [1,2,3, 14,15, 28,29,30,31,32,33,34,35,36,37,38,39,40]
#working = [4,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27]
gpio_pins_to_check = [6]#[4,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27]
# Set up the GPIO pin as an output

for gpio_pin in gpio_pins_to_check:
    try:
    	GPIO.setmode(GPIO.BCM)
    	GPIO.setup(gpio_pin, GPIO.OUT)
        # Turn on the GPIO pin
    	GPIO.output(gpio_pin, GPIO.LOW)
    	print("GPIO pin {} turned on with GPIO.LOW".format(gpio_pin))
    	# Wait for some time
    	time.sleep(time_to_check)
    	
    	GPIO.output(gpio_pin, GPIO.HIGH)
    	print("GPIO pin {} turned on with GPIO.HIGH".format(gpio_pin))
    	# Wait for some time
    	time.sleep(time_to_check)
    except:
        print("not working {}".format(gpio_pin))
    finally:
        # Clean up and reset the GPIO settings
    	GPIO.cleanup()
    	print("GPIO cleanup complete.")
