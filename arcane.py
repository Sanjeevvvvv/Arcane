import cv2
import mediapipe as mp
import pygame
import pyautogui
import math
import numpy as np
import multiprocessing as multi
from collections import deque
import time
import pystray
from PIL import Image, ImageDraw
import threading

# Screen dimensions
WIDTH, HEIGHT = 1920, 1080

# Colors
CYAN = (0, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TRANS_GRAY = (128, 128, 128, 128)
BLUE = (0, 150, 255)

# Global state
app_running = True
gesture_enabled = True

# --- Gesture Combos ---
class GestureComboDetector:
    def __init__(self):
        self.combo_buffer = deque(maxlen=5)
        self.combo_patterns = {
            ('peace', 'peace'): 'maximize_window',
            ('pinch', 'pinch'): 'screenshot',
            ('index', 'peace'): 'mute_toggle',
            ('peace', 'index'): 'brightness_up',
        }
        self.last_combo_time = 0
        self.combo_cooldown = 1.5
        self.combo_executed = set()
    
    def add_gesture(self, left_gesture, right_gesture):
        current_time = time.time()
        
        combo = (left_gesture, right_gesture)
        combo_key = f"{left_gesture}_{right_gesture}_{int(current_time)}"
        
        # Check for combo patterns
        if combo in self.combo_patterns:
            if current_time - self.last_combo_time > self.combo_cooldown:
                if combo_key not in self.combo_executed:
                    self.last_combo_time = current_time
                    self.combo_executed.add(combo_key)
                    self.execute_combo(self.combo_patterns[combo])
                    return self.combo_patterns[combo]
        
        # Clean old executed combos
        self.combo_executed = {k for k in self.combo_executed if int(k.split('_')[-1]) > current_time - 5}
        return None
    
    def execute_combo(self, combo_action):
        """Execute special combo actions"""
        if combo_action == 'maximize_window':
            pyautogui.hotkey('win', 'up')
        elif combo_action == 'screenshot':
            pyautogui.hotkey('win', 'shift', 's')
        elif combo_action == 'mute_toggle':
            pyautogui.press('volumemute')
        elif combo_action == 'brightness_up':
            # This varies by system - using volume as placeholder
            for _ in range(5):
                pyautogui.press('volumeup')

combo_detector = GestureComboDetector()

# --- Tutorial System ---
class TutorialSystem:
    def __init__(self):
        self.active = False
        self.step = 0
        self.steps = [
            {"gesture": "pinch_right", "text": "RIGHT HAND: Pinch to Play/Pause", "color": RED},
            {"gesture": "peace", "text": "RIGHT HAND: Peace Sign for Next Track", "color": YELLOW},
            {"gesture": "index", "text": "RIGHT HAND: Point for Previous Track", "color": PURPLE},
            {"gesture": "pinch_left", "text": "LEFT HAND: Pinch & Drag for Volume", "color": GREEN},
            {"gesture": "combo", "text": "BOTH HANDS: Try Peace+Peace for Maximize!", "color": CYAN},
        ]
        self.gesture_detected = False
        self.step_completion_time = 0
    
    def start(self):
        self.active = True
        self.step = 0
        self.gesture_detected = False
    
    def stop(self):
        self.active = False
    
    def check_gesture(self, detected_gesture):
        if not self.active or self.step >= len(self.steps):
            return False
        
        current_step = self.steps[self.step]
        if detected_gesture and current_step["gesture"] in str(detected_gesture):
            if not self.gesture_detected:
                self.gesture_detected = True
                self.step_completion_time = time.time()
                return True
        return False
    
    def next_step(self):
        if time.time() - self.step_completion_time > 2:
            self.step += 1
            self.gesture_detected = False
            if self.step >= len(self.steps):
                self.active = False
    
    def get_current_instruction(self):
        if self.active and self.step < len(self.steps):
            return self.steps[self.step]
        return None

tutorial = TutorialSystem()

# --- Gesture Recognition (GPU Accelerated) ---
def detect_gestures(hand_landmarks, handedness):
    if not hand_landmarks:
        return None, None

    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
    
    index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]

    pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y) * WIDTH

    is_peace = (index_tip.y < index_mcp.y and
                middle_tip.y < middle_mcp.y and
                ring_tip.y > ring_mcp.y and
                pinky_tip.y > pinky_mcp.y)

    is_index_pointer = (index_tip.y < index_mcp.y and
                        middle_tip.y > middle_mcp.y and
                        ring_tip.y > ring_mcp.y and
                        pinky_tip.y > pinky_mcp.y)

    hand = handedness.classification[0].label

    if hand == "Right":
        if pinch_dist < 50:
            return "playpause", hand
        elif is_peace:
            return "nexttrack", hand
        elif is_index_pointer:
            return "prevtrack", hand
    elif hand == "Left":
        if pinch_dist < 50:
            return "volume", hand
            
    return None, hand

# --- Vision Process (GPU Accelerated) ---
def vision_process(queue, control_queue):
    # Local variable for this process
    gesture_enabled = True
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # GPU Acceleration: Use model_complexity=1 for better performance
    hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1  # GPU optimized
    )

    gesture_state = {"Left": None, "Right": None}
    frame_skip = 2  # Process every 2nd frame for better FPS
    frame_count = 0

    while True:
        # Check control messages
        if not control_queue.empty():
            msg = control_queue.get()
            if msg == "stop":
                break
            elif msg == "toggle":
                gesture_enabled = not gesture_enabled

        success, image = cap.read()
        if not success:
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        detected_hands = []
        left_gesture = None
        right_gesture = None
        
        if results.multi_hand_landmarks and gesture_enabled:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                gesture, hand = detect_gestures(hand_landmarks, handedness)
                
                # Track gestures for combo detection
                if hand == "Left":
                    left_gesture = gesture
                elif hand == "Right":
                    right_gesture = gesture
                
                if gesture and gesture_state.get(hand) != gesture:
                    pyautogui.press(gesture)
                    gesture_state[hand] = gesture
                elif not gesture and gesture_state.get(hand) is not None:
                    gesture_state[hand] = None

                hand_data = {
                    'landmarks': [(lm.x * WIDTH, lm.y * HEIGHT) for lm in hand_landmarks.landmark],
                    'handedness': hand,
                    'gesture': gesture_state[hand]
                }
                detected_hands.append(hand_data)
        
        # Check for gesture combos
        if left_gesture and right_gesture:
            combo = combo_detector.add_gesture(left_gesture, right_gesture)
            if combo:
                detected_hands.append({'combo': combo})
        
        queue.put(detected_hands)

    cap.release()
    hands.close()

# --- System Tray Integration ---
def create_tray_icon():
    # Create icon image
    icon_image = Image.new('RGB', (64, 64), color=(0, 100, 200))
    draw = ImageDraw.Draw(icon_image)
    draw.ellipse([16, 16, 48, 48], fill=(0, 255, 255))
    draw.text((20, 20), "J", fill=(0, 0, 0))
    
    def on_quit(icon, item):
        global app_running
        app_running = False
        icon.stop()
    
    def on_toggle(icon, item):
        global gesture_enabled
        gesture_enabled = not gesture_enabled
        status = "✅ Enabled" if gesture_enabled else "❌ Disabled"
        print(f"Gestures {status}")
    
    def on_tutorial(icon, item):
        tutorial.start()
        print("📚 Tutorial Started!")
    
    menu = pystray.Menu(
        pystray.MenuItem("🎮 Toggle Gestures", on_toggle),
        pystray.MenuItem("📚 Start Tutorial", on_tutorial),
        pystray.MenuItem("❌ Quit", on_quit)
    )
    
    icon = pystray.Icon("JARVIS", icon_image, "JARVIS Media Control", menu)
    icon.run()

# --- Rendering Process ---
def rendering_process(queue, control_queue):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    pygame.display.set_caption("J.A.R.V.I.S Ultimate Media Control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    font_large = pygame.font.SysFont("Arial", 36, bold=True)
    font_small = pygame.font.SysFont("Arial", 18)

    # Volume control
    current_volume = 50
    target_volume = 50
    volume_bar_visible = False

    def draw_grid():
        for i in range(0, WIDTH, 50):
            pygame.draw.line(screen, (0, 50, 50), (i, 0), (i, HEIGHT))
        for i in range(0, HEIGHT, 50):
            pygame.draw.line(screen, (0, 50, 50), (0, i), (WIDTH, i))

    def draw_magic_ring(center_pos, color):
        for i in range(5):
            radius = 50 + i * 10
            alpha = 255 - i * 50
            temp_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, color + (alpha,), (radius, radius), radius, 2)
            screen.blit(temp_surface, (center_pos[0] - radius, center_pos[1] - radius))

    def draw_combo_effect(text):
        combo_surf = pygame.Surface((500, 120), pygame.SRCALPHA)
        combo_surf.fill((255, 215, 0, 200))
        combo_text = font_large.render(f"⚡ COMBO: {text.upper().replace('_', ' ')}", True, BLACK)
        combo_surf.blit(combo_text, (30, 40))
        screen.blit(combo_surf, (WIDTH // 2 - 250, HEIGHT // 2 - 60))

    def draw_hud(action, volume, gesture_status):
        hud_surface = pygame.Surface((420, 280), pygame.SRCALPHA)
        hud_surface.fill(TRANS_GRAY)
        
        title_text = font_large.render("J.A.R.V.I.S", True, CYAN)
        subtitle = font_small.render("Ultimate Media Control", True, WHITE)
        action_text = font.render(f"Action: {action}", True, WHITE)
        volume_text = font.render(f"Volume: {int(volume)}%", True, WHITE)
        
        hud_surface.blit(title_text, (20, 10))
        hud_surface.blit(subtitle, (20, 50))
        hud_surface.blit(action_text, (20, 90))
        hud_surface.blit(volume_text, (20, 125))
        
        # Status
        status = "🟢 ON" if gesture_status else "🔴 OFF"
        status_color = GREEN if gesture_status else RED
        status_text = font.render(f"Gestures: {status}", True, status_color)
        hud_surface.blit(status_text, (20, 160))
        
        # Instructions
        instructions = [
            "F1 = Tutorial | F2 = Toggle | ESC = Exit",
            "Tray Icon: Right-click for menu"
        ]
        y_pos = 200
        for inst in instructions:
            inst_text = font_small.render(inst, True, CYAN)
            hud_surface.blit(inst_text, (20, y_pos))
            y_pos += 25
        
        screen.blit(hud_surface, (20, 20))

    def draw_volume_bar(volume):
        bar_x, bar_y, bar_w, bar_h = WIDTH - 70, 250, 30, HEIGHT - 300
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_w, bar_h), 2)
        fill_h = bar_h * (volume / 100)
        pygame.draw.rect(screen, GREEN, (bar_x, bar_y + bar_h - fill_h, bar_w, fill_h))
        
        # Volume percentage text
        vol_text = font_small.render(f"{int(volume)}%", True, WHITE)
        screen.blit(vol_text, (bar_x - 10, bar_y - 30))

    def draw_tutorial(instruction):
        if instruction:
            tutorial_surf = pygame.Surface((WIDTH, 150), pygame.SRCALPHA)
            tutorial_surf.fill((0, 0, 0, 220))
            
            text = font_large.render(instruction["text"], True, instruction["color"])
            progress = font.render(f"Step {tutorial.step + 1}/{len(tutorial.steps)}", True, WHITE)
            hint = font_small.render("Complete the gesture to continue...", True, CYAN)
            
            text_rect = text.get_rect(center=(WIDTH // 2, 40))
            progress_rect = progress.get_rect(center=(WIDTH // 2, 85))
            hint_rect = hint.get_rect(center=(WIDTH // 2, 115))
            
            tutorial_surf.blit(text, text_rect)
            tutorial_surf.blit(progress, progress_rect)
            tutorial_surf.blit(hint, hint_rect)
            
            screen.blit(tutorial_surf, (0, HEIGHT - 150))

    def draw_gesture_confidence(hands_data):
        y_offset = HEIGHT // 2 - 100
        for hand in hands_data:
            if 'gesture' in hand and hand['gesture']:
                gesture_name = hand['gesture'].upper().replace('TRACK', ' TRACK')
                conf_text = font.render(f"{hand['handedness']}: {gesture_name}", True, CYAN)
                conf_bg = pygame.Surface((conf_text.get_width() + 20, 40), pygame.SRCALPHA)
                conf_bg.fill((0, 0, 0, 180))
                screen.blit(conf_bg, (WIDTH // 2 - conf_text.get_width() // 2 - 10, y_offset))
                screen.blit(conf_text, (WIDTH // 2 - conf_text.get_width() // 2, y_offset + 8))
                y_offset += 50

    combo_display_time = 0
    combo_text = ""
    local_gesture_enabled = True

    running = True
    while running and app_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    tutorial.start()
                    print("📚 Tutorial Mode Activated!")
                elif event.key == pygame.K_F2:
                    local_gesture_enabled = not local_gesture_enabled
                    control_queue.put("toggle")
                    print(f"🎮 Gestures: {'ON' if local_gesture_enabled else 'OFF'}")

        screen.fill(BLACK)
        draw_grid()

        hands_data = []
        while not queue.empty():
            hands_data = queue.get()

        current_action = "Idle"
        volume_bar_visible = False
        
        if hands_data:
            for hand in hands_data:
                # Check for combos
                if 'combo' in hand:
                    combo_text = hand['combo']
                    combo_display_time = time.time()
                    print(f"⚡ COMBO EXECUTED: {combo_text}")
                    continue
                
                center_pos = (int(np.mean([lm[0] for lm in hand['landmarks']])),
                              int(np.mean([lm[1] for lm in hand['landmarks']])))

                color = ORANGE
                if hand['gesture'] == 'playpause':
                    color = RED
                    current_action = "Play/Pause"
                    tutorial.check_gesture("pinch_right")
                elif hand['gesture'] == 'nexttrack':
                    color = YELLOW
                    current_action = "Next Track"
                    tutorial.check_gesture("peace")
                elif hand['gesture'] == 'prevtrack':
                    color = PURPLE
                    current_action = "Previous Track"
                    tutorial.check_gesture("index")
                elif hand['gesture'] == 'volume':
                    color = GREEN
                    current_action = "Volume Control"
                    volume_bar_visible = True
                    tutorial.check_gesture("pinch_left")
                    
                    hand_y = center_pos[1]
                    target_volume = np.interp(hand_y, [HEIGHT - 100, 100], [0, 100])
                    
                    vol_diff = (target_volume - current_volume) / 2
                    for _ in range(abs(int(vol_diff))):
                        pyautogui.press('volumeup' if vol_diff > 0 else 'volumedown')
                    current_volume = target_volume

                draw_magic_ring(center_pos, color)
                for lm in hand['landmarks']:
                    pygame.draw.circle(screen, CYAN, (int(lm[0]), int(lm[1])), 5)
            
            draw_gesture_confidence(hands_data)

        # Display combo effect
        if time.time() - combo_display_time < 2.5:
            draw_combo_effect(combo_text)

        draw_hud(current_action, current_volume, local_gesture_enabled)
        
        if volume_bar_visible:
            draw_volume_bar(current_volume)

        # Tutorial
        if tutorial.active:
            instruction = tutorial.get_current_instruction()
            draw_tutorial(instruction)
            if tutorial.gesture_detected:
                tutorial.next_step()

        # FPS counter
        fps = clock.get_fps()
        fps_text = font.render(f"FPS: {int(fps)}", True, GREEN if fps > 45 else YELLOW)
        screen.blit(fps_text, (WIDTH - 150, HEIGHT - 40))

        pygame.display.flip()
        clock.tick(60)

    control_queue.put("stop")
    pygame.quit()

# --- Main ---
if __name__ == "__main__":
    multi.set_start_method('spawn', force=True)
    
    q = multi.Queue()
    control_q = multi.Queue()
    
    # Start system tray in separate thread
    tray_thread = threading.Thread(target=create_tray_icon, daemon=True)
    tray_thread.start()
    
    # Start processes
    p_vision = multi.Process(target=vision_process, args=(q, control_q))
    
    p_vision.start()
    
    print("=" * 60)
    print("🚀 J.A.R.V.I.S ULTIMATE MEDIA CONTROL")
    print("=" * 60)
    print("✅ ALL 5 PREMIUM FEATURES ACTIVE:")
    print("   1. ⚡ GPU Acceleration - 60 FPS Target")
    print("   2. 🎮 Gesture Combos - Two-hand power moves")
    print("   3. 📚 Tutorial Mode - Press F1")
    print("   4. 🖥️  System Tray - Check your taskbar")
    print("   5. 🎵 Media Control - Works with any player")
    print("=" * 60)
    print("⌨️  CONTROLS:")
    print("   F1  = Start Tutorial")
    print("   F2  = Toggle Gestures On/Off")
    print("   ESC = Exit")
    print("=" * 60)
    print("🎮 GESTURE COMBOS (Both hands simultaneously):")
    print("   Peace + Peace = Maximize Window")
    print("   Pinch + Pinch = Screenshot")
    print("   Index + Peace = Mute Toggle")
    print("   Peace + Index = Brightness Up")
    print("=" * 60)
    
    rendering_process(q, control_q)
    
    p_vision.join()
    p_vision.terminate()
