"""
Word Flow Animations
Various animated text effects for terminal and data visualization.
"""

import time
import sys
import os
from itertools import cycle

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def clear_line():
    """Clear the current terminal line."""
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()


def typing_effect(text, delay=0.05, prefix=""):
    """Classic typing animation effect."""
    for char in text:
        sys.stdout.write(prefix + char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def typewriter_simulation(words, delay=0.1):
    """Simulate typewriter with word-by-word appearance."""
    for word in words:
        print(word, end=' ', flush=True)
        time.sleep(delay)
    print()


def word_carousel(words, iterations=3, delay=0.3):
    """Rotating word carousel effect."""
    word_cycle = cycle(words)
    for _ in range(len(words) * iterations):
        word = next(word_cycle)
        clear_line()
        print(f"► {word} ◄", end='', flush=True)
        time.sleep(delay)
    clear_line()


def waterfall_flow(words, delay=0.1):
    """Waterfall-style word flow from top to bottom."""
    if not HAS_NUMPY:
        print("numpy not available for waterfall effect")
        return
        
    height, width = os.get_terminal_size()
    width = min(width, 80)
    
    for frame in range(len(words)):
        clear_line()
        lines = []
        for i in range(min(5, len(words))):
            idx = (frame - i) % len(words)
            indent = " " * (i * 3)
            lines.append(f"{indent}{words[idx]}")
        print('\n'.join(lines))
        time.sleep(delay)


def matrix_style(words, iterations=10, delay=0.1):
    """Matrix-style falling word effect."""
    if not HAS_NUMPY:
        print("numpy not available for matrix effect")
        return
        
    height, width = os.get_terminal_size()
    width = min(width, 80)
    
    cols = min(10, len(words))
    positions = np.zeros(cols, dtype=int)
    speeds = np.random.randint(1, 4, cols)
    
    for frame in range(iterations):
        clear_line()
        grid = [[' ' for _ in range(width)] for _ in range(10)]
        
        for i in range(cols):
            word_idx = (frame + i) % len(words)
            word = words[word_idx][:10]
            col = i * (width // cols)
            row = positions[i]
            if row < 10 and col + len(word) < width:
                for j, c in enumerate(word):
                    if col + j < width:
                        grid[row][col + j] = c
            positions[i] = (positions[i] + speeds[i]) % 15
        
        for row in grid:
            print(''.join(row))
        time.sleep(delay)


def pulse_glow(words, iterations=5, delay=0.5):
    """Pulsing/glowing word effect with borders."""
    borders = ['─' * 40, '═' * 40, '┄' * 40, '▀' * 40, '▄' * 40]
    border_cycle = cycle(borders)
    
    for _ in range(iterations):
        border = next(border_cycle)
        word = words[_ % len(words)]
        
        clear_line()
        print(f"╭{border}╮")
        print(f"│{word:^40}│")
        print(f"╰{border}╯")
        time.sleep(delay)


def bounce_effect(words, iterations=4):
    """Bouncing word animation."""
    for word in words[:5]:
        for i in range(3):
            indent = " " * (10 - i * 3 if i < 3 else i * 3 - 9)
            clear_line()
            print(f"{indent}○ {word}")
            time.sleep(0.1)
        for i in range(3, -1, -1):
            indent = " " * (10 - i * 3 if i < 3 else i * 3 - 9)
            clear_line()
            print(f"{indent}○ {word}")
            time.sleep(0.1)


def rainbow_flow(words, iterations=2):
    """Colorful word flow (ANSI colors)."""
    colors = [
        '\033[91m',  # Red
        '\033[93m',  # Yellow
        '\033[92m',  # Green
        '\033[96m',  # Cyan
        '\033[94m',  # Blue
        '\033[95m',  # Magenta
    ]
    reset = '\033[0m'
    
    for _ in range(iterations):
        for i, word in enumerate(words):
            color = colors[i % len(colors)]
            clear_line()
            print(f"{color}◆ {word}{reset}")
            time.sleep(0.2)


def spotlight_reveal(words, delay=0.3):
    """Text reveal with spotlight/dim effect."""
    for word in words:
        for i in range(5):
            clear_line()
            dim = '\033[2m' if i < 3 else ''
            bright = '\033[1m' if i >= 3 else ''
            print(f"{dim}{bright}▓ {word} ▓{reset if i >= 3 else ''}")
            time.sleep(0.1)
        print(word)
        time.sleep(delay)


def slide_in(words, delay=0.15):
    """Words sliding in from left."""
    for word in words:
        for width in range(40, -1, -2):
            clear_line()
            print(f"{' ' * width}→ {word}")
            time.sleep(0.02)
        print(f"{' ' * 40}→ {word}")
        time.sleep(delay)


def wave_flow(words, iterations=3, delay=0.1):
    """Wave-like word flow."""
    wave_chars = ['~', '≈', '≋', '∿', '⍨']
    
    for _ in range(iterations):
        for i, word in enumerate(words):
            wave = wave_chars[i % len(wave_chars)]
            clear_line()
            print(f"{word} {wave}")
            time.sleep(delay)


def typewriter_poem():
    """Full typewriter poem animation."""
    lines = [
        "In circuits deep, the algorithms dream,",
        "Of paths through mazes, a navigation stream.",
        "Bio-inspired wisdom guides the way,",
        "Through obstacles and challenges, day by day.",
        "",
        "The neural fires, the fuzzy logic glow,",
        "Particle swarms, ant colonies they flow.",
        "Each algorithm a creature of grace,",
        "Finding the goal, securing the space.",
    ]
    
    for line in lines:
        typing_effect(line, delay=0.05)
        time.sleep(0.3)


def flow_intro(title, words):
    """Complete word flow introduction."""
    print("\n" + "=" * 60)
    print("  " + title.upper())
    print("=" * 60 + "\n")
    
    time.sleep(0.5)
    print("Loading words flow...")
    time.sleep(0.3)
    
    print("\n--- Typewriter Effect ---")
    typing_effect("Bio-Inspired Robotics Navigation Simulation", delay=0.03)
    
    print("\n--- Rainbow Flow ---")
    rainbow_flow(words[:6], iterations=1)
    
    print("\n--- Wave Flow ---")
    wave_flow(words[:8], iterations=2)
    
    print("\n--- Carousel ---")
    word_carousel(words, iterations=2, delay=0.2)
    
    print("\n" + "=" * 60)
    print("  Animation Complete!")
    print("=" * 60 + "\n")


def demo_all():
    """Run all word flow effects as a demo."""
    words = [
        "Navigation", "Algorithms", "Robotics", "Bio-Inspired",
        "Fuzzy Logic", "Neural Networks", "Particle Swarm",
        "Ant Colony", "Q-Learning", "Genetic", "Evolution",
        "Swarm Intelligence", "Path Planning", "Optimization"
    ]
    
    flow_intro("Word Flow Animations", words)
    
    print("--- Pulse Glow ---")
    pulse_glow(words[:4], iterations=3)
    
    print("\n--- Slide In ---")
    slide_in(words[:6], delay=0.2)
    
    print("\n--- Wave Flow ---")
    wave_flow(words, iterations=2)
    
    print("\n--- Matrix Style ---")
    matrix_style(words[:8], iterations=8)
    
    print("\n--- Waterfall ---")
    waterfall_flow(words, delay=0.15)
    
    print("\n--- Typewriter Poem ---")
    typewriter_poem()
    
    print("\n✓ All word flow effects demonstrated!")


if __name__ == "__main__":
    demo_all()
