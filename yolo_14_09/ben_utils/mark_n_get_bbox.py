path  = r'Z:\yolo_14_09\data\Baloons_DATA\retagged_7_8_22\rectangles_extra_images\extra_images_10_7_23\extra_images_10_7_23_Ibdis_21-05-23_7A_Img_12223.png'

# The following code will take snapshot of the region selected
# by mouse and confirming it by pressing letter w

import cv2
import os
import mss
import mss.tools
from pynput import mouse
from pynput import keyboard
global myKey
myKey=''
global xm,ym
xm,ym=0,0
def on_move(x, y):
    global xm,ym
    xm,ym=x,y
    print('Pointer moved to {0}'.format((xm, ym)))

def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    if not pressed:
        # Stop listener
        return False

def on_scroll(x, y, dx, dy):
    print('Scrolled {0} at {1}'.format(
        'down' if dy < 0 else 'up',
        (x, y)))
def on_press(key):
    global myKey

    try:
        print('alphanumeric key {0} pressed'.format( key.char))
        myKey=key
    except AttributeError:
        print('special key {0} pressed'.format(key))
        myKey=key
def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False

listener = mouse.Listener(
    on_move=on_move,
    on_click=on_click,
    on_scroll=on_scroll)


listenerk = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)

mousePoints = []
cleanScreen=False
drawing = False
writer = None
with mss.mss() as sct:
    sct.shot()

global rectDone
rectDone=False
drawing =False

drawing = False
global x1, y1, x2,y2
x1,y1,x2,y2=0,0,0,0
def draw_rect(event, x, y, flags, param):
    global x1, y1, drawing,  num, img, img2,x2,y2
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            a, b = x, y
            if a != x & b != y:
                img = img2.copy()

                cv2.rectangle(img, (x1,y1),(x,y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        num += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        x2 = x
        y2= y


key=ord('a')
img = cv2.imread(path)
img2=img.copy()
cv2.namedWindow("main", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("main", draw_rect)
cv2.setWindowProperty("main", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
num = 0
# PRESS w to confirm save selected bounded box
while key!=ord('w'):
    cv2.imshow("main",img)
    key = cv2.waitKey(1)&0xFF
print('Here are points:',x1,y1,x2,y2)
if key==ord('w'):
    cv2.imwrite('snap.png',img2[y1:y2,x1:x2])
    cv2.destroyAllWindows()
    print('Saved as snap.png')
    os.remove('monitor-1.png')
