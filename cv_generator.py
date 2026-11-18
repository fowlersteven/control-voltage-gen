"""
Multi-channel CV generator for RP2040 (MicroPython)

8 PWM outputs, 4 pots. Each pot controls the speed/timing of TWO outputs.
Every output is a plain function that returns a value from 0.0 to 1.0
(0.5 is the "zero" point for AC-style signals). Put an RC low-pass filter
on each PWM pin to turn it into a real control voltage.

To add a new CV type:
  1. Write a function with the signature  my_cv(ch, dt, pot)  that returns 0.0-1.0
  2. Add a line for it in the CHANNELS table below.

Save as main.py on the board to run at power-up.
"""

import random
import time
from math import sin, cos, sqrt, pi
from machine import Pin, PWM, ADC


# =============================================================================
# SETTINGS
# =============================================================================

# --- Hardware ---
POT_PINS = [26, 27, 28, 29]   # ADC0..ADC3. (GP29 is wired to VSYS on a stock Pico.)
PWM_FREQ_HZ = 50000           # PWM carrier. Higher = easier filtering, fewer bits (~11 bits at 50 kHz)

# --- Timing ---
LOOP_DELAY_MS = 1             # pause between output updates
POT_READ_INTERVAL_MS = 20     # how often the pots are read
POT_SMOOTHING = 0.2           # 0-1. Smaller = smoother pot reads but slower response

# --- LFO-style generators (pot sets frequency) ---
LFO_MIN_HZ = 0.1
LFO_MAX_HZ = 5.0

# --- Weird math generators ---
GOLDEN_RATIO = 1.618          # frequency ratio for the quasi-periodic beating
FM_RATIO = 0.37               # modulator frequency / carrier frequency
FM_DEPTH = 3.0                # how hard the modulator pushes the carrier (radians)

# --- Random walk ---
WALK_SIZE = 1.0               # typical wander per cycle (1.0 = roughly 0.3 of full range)

# --- Burst A: pulse train, then rest (pot sets rest length) ---
PULSES_PER_BURST = 4
PULSE_TIME_S = 0.04           # length of each ON and each OFF segment
PULSE_REST_MIN_S = 0.25
PULSE_REST_MAX_S = 10.0

# --- Burst B: random-voltage step sequence (pot sets time between bursts) ---
STEPS_PER_BURST = 8
STEP_TIME_S = 0.06            # how long each random voltage is held
STEP_REST_LEVEL = 0.5         # output between bursts (0.5 = AC zero)
STEP_GAP_MIN_S = 0.5
STEP_GAP_MAX_S = 20.0


# =============================================================================
# HELPERS
# =============================================================================

def pot_to_range(pot, low, high):
    """Map a pot (0.0-1.0) to low..high exponentially, so it feels natural."""
    return low * (high / low) ** pot


def advance(phase, freq, dt):
    """Move a 0.0-1.0 phase forward by one time step, wrapping at 1.0."""
    return (phase + freq * dt) % 1.0


# =============================================================================
# GENERATORS
# Each takes (ch, dt, pot):
#   ch  = this channel's state dict (phase, timer, step, ...)
#   dt  = seconds since the last call
#   pot = smoothed pot position, 0.0-1.0
# and returns an output level from 0.0 to 1.0.
# =============================================================================

# ---- Pseudorandom --------------------------------------------------------

def random_glide(ch, dt, pot):
    """Picks a new random target every cycle and glides to it with a cosine curve."""
    freq = pot_to_range(pot, LFO_MIN_HZ, LFO_MAX_HZ)
    old_phase = ch["phase"]
    ch["phase"] = advance(old_phase, freq, dt)
    if ch["phase"] < old_phase:             # cycle finished -> choose a new target
        ch["start"] = ch["target"]
        ch["target"] = random.random()
    blend = 0.5 - 0.5 * cos(pi * ch["phase"])   # smooth 0 -> 1 over the cycle
    return ch["start"] + (ch["target"] - ch["start"]) * blend


def random_walk(ch, dt, pot):
    """Wanders up and down in small random steps, bouncing off 0 and 1."""
    freq = pot_to_range(pot, LFO_MIN_HZ, LFO_MAX_HZ)
    # sqrt keeps the wandering speed the same no matter how fast the loop runs
    step = (random.random() - 0.5) * WALK_SIZE * sqrt(freq * dt)
    value = ch["value"] + step
    if value > 1.0:
        value = 2.0 - value
    elif value < 0.0:
        value = -value
    ch["value"] = value
    return value


# ---- Basic waveforms -----------------------------------------------------

def sine(ch, dt, pot):
    freq = pot_to_range(pot, LFO_MIN_HZ, LFO_MAX_HZ)
    ch["phase"] = advance(ch["phase"], freq, dt)
    return 0.5 + 0.5 * sin(2 * pi * ch["phase"])


def sawtooth(ch, dt, pot):
    freq = pot_to_range(pot, LFO_MIN_HZ, LFO_MAX_HZ)
    ch["phase"] = advance(ch["phase"], freq, dt)
    return ch["phase"]


# ---- Weird math ----------------------------------------------------------

def quasi_periodic(ch, dt, pot):
    """Two sines at an irrational ratio. Never quite repeats."""
    freq = pot_to_range(pot, LFO_MIN_HZ, LFO_MAX_HZ)
    ch["phase"] = advance(ch["phase"], freq, dt)
    ch["phase2"] = advance(ch["phase2"], freq * GOLDEN_RATIO, dt)
    total = sin(2 * pi * ch["phase"]) + sin(2 * pi * ch["phase2"])   # -2..2
    return 0.5 + 0.25 * total


def fm_wobble(ch, dt, pot):
    """A sine whose phase is pushed around by a slower sine (frequency modulation)."""
    freq = pot_to_range(pot, LFO_MIN_HZ, LFO_MAX_HZ)
    ch["phase"] = advance(ch["phase"], freq, dt)
    ch["phase2"] = advance(ch["phase2"], freq * FM_RATIO, dt)
    modulator = sin(2 * pi * ch["phase2"])
    return 0.5 + 0.5 * sin(2 * pi * ch["phase"] + FM_DEPTH * modulator)


# ---- Bursts --------------------------------------------------------------

def burst_pulses(ch, dt, pot):
    """A train of quick on/off pulses, then a rest. Pot sets the rest length."""
    rest_time = pot_to_range(pot, PULSE_REST_MIN_S, PULSE_REST_MAX_S)
    train_steps = PULSES_PER_BURST * 2          # each pulse = one ON step + one OFF step

    ch["timer"] += dt
    if ch["step"] < train_steps:                # currently in the pulse train
        if ch["timer"] >= PULSE_TIME_S:
            ch["timer"] -= PULSE_TIME_S
            ch["step"] += 1
    elif ch["timer"] >= rest_time:              # rest is over, start a new burst
        ch["timer"] = 0.0
        ch["step"] = 0

    if ch["step"] < train_steps and ch["step"] % 2 == 0:
        return 1.0                              # even steps are ON
    return 0.0


def burst_random_steps(ch, dt, pot):
    """A burst of random held voltages, then a gap. Pot sets the gap length."""
    gap_time = pot_to_range(pot, STEP_GAP_MIN_S, STEP_GAP_MAX_S)

    ch["timer"] += dt
    if ch["step"] < STEPS_PER_BURST:            # currently stepping through a burst
        if ch["timer"] >= STEP_TIME_S:
            ch["timer"] -= STEP_TIME_S
            ch["step"] += 1
            if ch["step"] < STEPS_PER_BURST:
                ch["value"] = random.random()   # next random voltage
            else:
                ch["value"] = STEP_REST_LEVEL   # burst finished
    elif ch["timer"] >= gap_time:               # gap is over, start a new burst
        ch["timer"] = 0.0
        ch["step"] = 0
        ch["value"] = random.random()
    return ch["value"]


# =============================================================================
# CHANNEL TABLE: (PWM pin, pot index, generator function)
# Pot index is the position in POT_PINS. Two channels share each pot.
# =============================================================================

CHANNELS = [
    # Pot 0: pseudorandom
    (0, 0, random_glide),
    (1, 0, random_walk),
    # Pot 1: waveforms
    (2, 1, sine),
    (3, 1, sawtooth),
    # Pot 2: bursts
    (4, 2, burst_pulses),
    (5, 2, burst_random_steps),
    # Pot 3: weird math
    (6, 3, quasi_periodic),
    (7, 3, fm_wobble),
]


def make_channel(pwm_pin, pot_index, generator):
    """Build the state dict for one output. Every generator uses what it needs."""
    return {
        "gen": generator,
        "pot": pot_index,
        "pwm": PWM(Pin(pwm_pin), freq=PWM_FREQ_HZ, duty_u16=32768),
        "phase": 0.0,
        "phase2": 0.0,
        "timer": 0.0,
        "step": 0,
        "start": random.random(),
        "target": random.random(),
        "value": random.random(),
    }


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    pots = [ADC(Pin(p)) for p in POT_PINS]
    pot_values = [adc.read_u16() / 65535 for adc in pots]   # start where the pots are
    channels = [make_channel(*c) for c in CHANNELS]

    pot_interval_us = POT_READ_INTERVAL_MS * 1000
    last_time = time.ticks_us()
    next_pot_read = last_time

    while True:
        now = time.ticks_us()
        dt = time.ticks_diff(now, last_time) / 1000000
        last_time = now

        # Read and smooth the pots every so often
        if time.ticks_diff(now, next_pot_read) >= 0:
            next_pot_read = time.ticks_add(now, pot_interval_us)
            for i, adc in enumerate(pots):
                raw = adc.read_u16() / 65535
                pot_values[i] += (raw - pot_values[i]) * POT_SMOOTHING

        # Update every output
        for ch in channels:
            level = ch["gen"](ch, dt, pot_values[ch["pot"]])
            level = min(1.0, max(0.0, level))
            ch["pwm"].duty_u16(int(level * 65535))

        time.sleep_ms(LOOP_DELAY_MS)


main()
