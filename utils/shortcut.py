import pyautogui

cursor_move_distance = 5
cursor_move_duration = 0

click_duration = 0.5
def call(x):
    pyautogui.write("hajaha")
    pyautogui.press('enter')

gesture_to_shortcuts = {
    'call': call,
    'dislike': lambda x: pyautogui.move(0, cursor_move_distance * x, duration=cursor_move_duration),
    'fist': None,
    'like': lambda x: pyautogui.move(0, -cursor_move_distance * x, duration=cursor_move_duration),
    'ok': lambda x: pyautogui.click(duration=click_duration),
    'one': lambda x: pyautogui.move(-cursor_move_distance * x, 0, duration=cursor_move_duration),
    'palm': lambda x: pyautogui.rightClick(duration=click_duration),
    'peace': lambda x: pyautogui.move(cursor_move_distance * x, 0, duration=cursor_move_duration),
    'rock': None,
    'three': lambda x: pyautogui.scroll(-30 * x),
    'two_up': lambda x: pyautogui.move(cursor_move_distance, 0, duration=cursor_move_duration),
    'three2': lambda x: pyautogui.scroll(30 * x)
}


# gesture_to_shortcuts = {
#     'call': None,
#     'dislike': lambda x: pyautogui.scroll(-500),
#     'fist': None,
#     'like': lambda x: pyautogui.scroll(500),
#     'ok': None,
#     'one': None,
#     'palm': None,
#     'peace': None,
#     'rock': None,
#     'three': None,
#     'two_up': None,
#     'three2': None
# }
