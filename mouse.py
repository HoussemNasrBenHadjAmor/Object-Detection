import pyautogui
import time

def move_and_click_mouse():
    while True:
        # Move the mouse slightly
        pyautogui.moveRel(0, 10, duration=0.5)  # Move mouse 10 pixels down
        time.sleep(1)  # Wait for 1 second
        pyautogui.moveRel(0, -10, duration=0.5)  # Move mouse back up
        
        # Perform a click
        pyautogui.click()
        
        # Wait for 59 seconds to complete the 1-minute interval
        time.sleep(59)

if __name__ == "__main__":  # Corrected line
    move_and_click_mouse()