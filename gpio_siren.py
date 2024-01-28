import Jetson.GPIO as GPIO
import time

def ActivateSiren():
    # Set the GPIO mode to BCM
    GPIO.setmode(GPIO.BCM)

    # Define the GPIO pin number
    gpio_pin = 6

    # Set up the GPIO pin as an output
    GPIO.setup(gpio_pin, GPIO.OUT)

    try:
        # Turn on the GPIO pin
        GPIO.output(gpio_pin, GPIO.HIGH)
        print("GPIO pin {} turned on.".format(gpio_pin))
        # Wait for some time
        time.sleep(2)

    finally:
        # Clean up and reset the GPIO settings
        GPIO.cleanup()
        print("GPIO cleanup complete.")

if __name__ == "__main__":
    ActivateSiren()
