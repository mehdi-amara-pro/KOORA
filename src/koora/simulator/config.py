"""
Tunable constants for the match simulator.

Units: metres, seconds, m/s. Pitch coordinates follow the contracts:
105 x 68 m, origin at the home team's defensive left corner in period 1.
Change values here to tune the simulation; nothing else should hard-code them.
"""


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a single number. Much faster than np.clip for scalars."""
    return lo if value < lo else hi if value > hi else value


# --- Pitch ---------------------------------------------------------------
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
CENTRE_X = PITCH_LENGTH / 2
CENTRE_Y = PITCH_WIDTH / 2
GOAL_HALF_WIDTH = 3.66
GOAL_HEIGHT = 2.44
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_HALF_WIDTH = 20.16
CENTRE_CIRCLE_RADIUS = 9.15

# --- Time ----------------------------------------------------------------
TICK_HZ = 25
DT = 1.0 / TICK_HZ

# --- Ball physics ----------------------------------------------------------
GRAVITY = 9.81
# Air drag: a = -AIR_DRAG * |v| * v, where AIR_DRAG = 0.5 * rho * Cd * A / m
# rho = 1.2 kg/m3, Cd = 0.25, A = pi * 0.11^2 m2, m = 0.43 kg  ->  ~0.013 per metre
AIR_DRAG = 0.013
ROLLING_DECEL = 1.6  # m/s2, grass resistance on a rolling ball
BOUNCE_RESTITUTION = 0.55  # share of vertical speed kept after a bounce
BOUNCE_FRICTION = 0.8  # share of horizontal speed kept after a bounce
BALL_SUBSTEPS = 4  # physics substeps per tick (drag is stiff at high speed)

# --- Players -------------------------------------------------------------
REACTION_TIME = 0.3  # s before a player starts moving towards a new point
ACCELERATION = 3.5  # m/s2
KEEPER_DIVE_ACCELERATION = 12.0  # m/s2, explosive lateral dive
JOG_SPEED = 3.6  # m/s, positional movement
RUN_SPEED = 6.0  # m/s, recovering position from far away
ARRIVAL_GAIN = 0.8  # desired speed = gain * distance to target, so players ease into position
DEADBAND = 0.8  # m, close enough: stand still instead of fidgeting
SEPARATION = 0.9  # m, players closer than this get pushed apart
STAMINA_DRAIN = 0.0008  # per second per m/s above SPRINT_DRAIN_FROM
SPRINT_DRAIN_FROM = 5.5  # m/s
STAMINA_RECOVERY = 0.002  # per second while walking
STAMINA_SPEED_FLOOR = 0.88  # tired players keep at least 88% of top speed

# --- Ball control ----------------------------------------------------------
CONTROL_RADIUS = 1.0  # m, outfield player can control the ball within this
KEEPER_CONTROL_RADIUS = 1.8
CONTROL_MAX_HEIGHT = 1.9  # m, chest or head height
KEEPER_MAX_HEIGHT = 2.6
CONTROL_MAX_SPEED = 26.0  # m/s, faster balls cannot be controlled by outfield players
CLEAN_CONTROL_SPEED = 14.0  # m/s, above this interceptions may only deflect the ball
KICK_COOLDOWN = 0.35  # s, a kicker cannot touch their own kick straight away
DRIBBLE_OFFSET = 0.6  # m, ball distance in front of a dribbling player
DRIBBLE_SPEED_FACTOR = 0.85  # dribbling players run at 85% of top speed

# --- Duels -----------------------------------------------------------------
PRESSURE_START = 2.5  # m, an opponent this close starts a pressure episode
PRESSURE_END = 3.5  # m
MIN_PRESSURE_DURATION = 0.4  # s, shorter episodes are not reported
PRESS_DISTANCE = 1.2  # m, how tight the presser gets (goal side of the carrier)
TACKLE_RANGE = 1.4  # m
TACKLE_RATE = 0.2  # tackle attempts per second while in range
BLOCK_RANGE = 1.6  # m, opponents this close to a kick can block it
BLOCK_CHANCE = 0.25  # chance each of them gets a touch
TACKLE_COOLDOWN = 1.0  # s between attempts by the same player
FOUL_SHARE = 0.55  # share of failed tackles that are fouls
YELLOW_CARD_SHARE = 0.1

# --- Decisions -------------------------------------------------------------
DECISION_TEMPERATURE = 0.006  # softmax temperature: lower = always the best option
MIN_SHOT_X = 75.0  # team frame: no shots from further out than ~30 m
SHOT_VALUE_FACTOR = 0.9  # < 1 makes players prefer working a better chance over shooting
POSSESSION_VALUE = 0.03  # what keeping the ball is worth by itself (future chances), in goals
UNDER_PRESSURE_DISTANCE = 3.0  # m
FIRST_TOUCH_TIME = (0.6, 1.6)  # s, range of time on the ball before the first decision
CARRY_TIME = (0.5, 1.0)  # s, how long a carry lasts before deciding again

# --- Restarts (seconds before play resumes) -------------------------------
# Real matches have the ball out of play for ~1/3 of the time; these are shortened for watchability.
KICKOFF_DELAY = 1.0
GOAL_CELEBRATION = 12.0
THROW_IN_DELAY = 5.0
CORNER_DELAY = 9.0
GOAL_KICK_DELAY = 7.0
FREE_KICK_DELAY = 7.0

# --- Reporting ---------------------------------------------------------------
REAL_BALL_IN_PLAY_MINUTES = 57.0  # typical top-flight match, used to compare event counts fairly
