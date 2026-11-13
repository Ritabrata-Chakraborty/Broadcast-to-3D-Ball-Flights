"""Project parameters: paths, the LP match, flight selection, folds, model bounds and evaluation."""
from pathlib import Path

# ---------------------------------------------------------------- Paths

REPO_ROOT = Path(__file__).resolve().parent
BALL3D_ROOT = REPO_ROOT / "external" / "ball3d"          # Grad et al.'s code and data (read-only)
DATA_ROOT = BALL3D_ROOT / "data"
WEIGHTS_ROOT = REPO_ROOT / "weights"
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"              # PnLCalib clone

# Every stage writes below OUT_ROOT, one folder per stage (see README).
OUT_ROOT = REPO_ROOT / "outputs"
FLIGHTS_DIR = OUT_ROOT / "flights"        # selected flights, splits and oracle fits
EXTRACT_DIR = OUT_ROOT / "extract"        # YOLO boxes and PnLCalib cameras per view and half
HEADS_DIR = OUT_ROOT / "heads"            # fine-tuned YOLO and PnLCalib heads and their data
FOLDS_DIR = OUT_ROOT / "folds"            # per-fold view-flight datasets per input setting
EVAL_ROOT = OUT_ROOT / "eval"             # test sets, runs, predictions, selection, report
EVAL_DATA = EVAL_ROOT / "data"
FIGURES_DIR = OUT_ROOT / "figures"
CALIB_ROOT = OUT_ROOT / "calibration"     # PnLCalib MV vs SoccerMaster study (supplement)

# ---------------------------------------------------------------- The LP match of Grad et al.

LP = "LP"
HALVES = ("half_1", "half_2")
DM_PER_M = 10.0                           # Grad et al. store lengths in decimetres
# Broadcast frame = static-view frame + offset, measured on the 3D ground truth per half.
LP_VIEW_OFFSETS = {
    "half_1": {"camera01": -1496, "camera02": 3, "camera03": 34, "camera04": 3, "camera05": 6,
               "camera06": 23},
    "half_2": {"camera01": 3, "camera02": 2, "camera03": 58, "camera04": 2, "camera05": 5,
               "camera06": -3},
}
# Pitch centre (kickoff ball position, GT frame 0 of both halves) in the dataset frame.
LP_PITCH_CENTRE_DM = (575.0, 391.0)
PITCH_CENTRE_DM = {"LP": LP_PITCH_CENTRE_DM, "ISSIA-3D": (575.0, 390.0)}
# The released pitch points (dev/df_merged_ball_player.csv) of camera01 and camera05 are given in
# a frame turned 180 degrees about the pitch centre; Grad et al.'s GT-input runs turn them back.
LP_ROTATED_DEV_VIEWS = (1, 5)

# ---------------------------------------------------------------- Flights

FPS = 25.0
G_MPS2 = 9.806
SUBSTEPS = 1                              # RK4 steps per video frame
MIN_FRAMES = 8
MAX_FRAMES = 150
MIN_APEX_M = 1.0
MAX_SPEED_MPS = 40.0                      # mean path speed; above this the GT is glitched
MIN_COVERAGE = 0.8                        # share of a flight's frames with 3D GT
MIN_VISIBLE = 0.5                         # share of a flight's frames inside the image, per view
VIEW_MATCH_FRAMES = 3                     # tolerance matching a static view's contact to broadcast
START_BINS_M = (0.5, 1.3)                 # start-height strata: ground, thigh or chest, head
IMAGE_WIDTH_PX = 1920.0                   # all pixel quantities are expressed at this width
OUTLIER_PX = 150.0                        # dataset detections this far from projected GT dropped
# Half-based splits of the flight table; the head fine-tuning folds use them ("a" trains on
# half 1 and is applied to half 2, "b" the reverse).
HALF_FOLDS = {"a": "half_1", "b": "half_2"}
VAL_SHARE = 0.2
SPLIT_GAP_FRAMES = 50
MIN_SPIN = 0.004                          # oracle coefficient size at which a sign is scored

# ---------------------------------------------------------------- Flight model bounds

K3_MAX = 0.06                             # quadratic drag, 1/m
KL_KS_MAX = 0.03                          # lift and sidespin, 1/m
Z0_MAX_M = 4.0
MAX_RAY_M = 200.0
MAX_SKIPPED_STEPS = 50
MAX_PRED_SPEED_MPS = 60.0

# ---------------------------------------------------------------- Folds and evaluation

VIEWS = (0, 1, 2, 3, 4, 5)                # broadcast and static cameras 01-05; camera 06 unused
BLOCK_GAP_FRAMES = 50
# Fold: (test block, validation block, train blocks). A = half 1, B = half 2; 1 early, 2 late.
FOLDS = {"1": ("A1", "B1", ("A2", "B2")), "2": ("A2", "B2", ("A1", "B1")),
         "3": ("B1", "A2", ("A1", "B2")), "4": ("B2", "A1", ("A2", "B1"))}
HIGH_CONTACT_M = 0.3                      # GT ball-centre height at which a contact counts as high
BALL_RADIUS_M = 0.11
BALL_DIAMETER_M = 0.22
CONTACT_SEARCH_FRAMES = 2                 # how far from a contact frame the GT height may come
ANCHOR_SEARCH_FRAMES = 4                  # nearest detection to a contact frame used for an anchor
CAP_M = 10.0                              # per-frame error cap; a missing frame counts as this
MIN_PART_FRAMES = 17                      # partial-flight population: >= 6 frames seen at 30 %
# Real-input track per fold, chosen on validation: Track-Phys in folds 1-3, Track-CV in fold 4.
# The same choice sets the track-gap statistics of the training corruption.
TRACK_SOURCE = {"1": "phys", "2": "phys", "3": "phys", "4": "cv"}

# ---------------------------------------------------------------- Calibration study (supplement)

ISSIA_PNL_CAMERAS = (1, 2, 5, 6)
ISSIA_VIDEO = "ISSIA-2D/Sequences/Film Role-0 ID-{camera} T-2 m00s00-000-m00s00-185.avi"
ISSIA_MIRRORED_CAMERAS = (2, 6)           # AVI is mirrored relative to the annotations and cameras
SN_DATA_ROOT = REPO_ROOT / "data" / "soccernet"
SOCCERMASTER_ROOT = REPO_ROOT / "external" / "SoccerMaster" / "codes" / "SoccerMaster"
SOCCERMASTER_WEIGHTS = WEIGHTS_ROOT / "soccermaster"
SIGLIP2_WEIGHTS = WEIGHTS_ROOT / "siglip2" / "siglip2-large-patch16-512"
WC14_ROOT = REPO_ROOT / "data" / "wc14" / "test"
TSWC_ROOT = REPO_ROOT / "data" / "tswc" / "TS-WorldCup"
# Published PnLCalib weights per variant; "pnlcalib" is the generic single-view pair.
CALIB_PNL_WEIGHTS = {"pnlcalib": ("SV_kp", "SV_lines"), "pnlcalib_mv": ("MV_kp", "MV_lines"),
                     "pnlcalib_ft_wc14": ("SV_FT_WC14_kp", "SV_FT_WC14_lines"),
                     "pnlcalib_ft_tswc": ("SV_FT_TSWC_kp", "SV_FT_TSWC_lines")}
# Heatmap peak separation (keypoints, lines): SoccerMaster's own metrics code, or PnLCalib's.
CALIB_SM_PEAKS = {"soccermaster": (15, 10), "soccermaster_d1": (1, 1)}
CALIB_MODELS = ("pnlcalib", "pnlcalib_mv", "soccermaster", "soccermaster_d1")
CALIB_VARIANTS = (*CALIB_PNL_WEIGHTS, *CALIB_SM_PEAKS)
CALIB_NET_WH = {**{name: (960, 540) for name in CALIB_PNL_WEIGHTS},
                **{name: (512, 512) for name in CALIB_SM_PEAKS}}
CALIB_SEED = 0
CALIB_EVAL_WH = (960, 540)                # official SoccerNet evaluation resolution
CALIB_JAC_PX = (5, 10, 20)                # official thresholds at that resolution
CALIB_MAX_REP_ERR_PX = 100.0              # at 960 wide; PnLCalib's SN22 benchmark script
CALIB_KP_GRID = (0.05, 0.0966, 0.1611, 0.25, 0.3434, 0.5)
CALIB_LINE_GRID = (0.1, 0.2, 0.3441, 0.5, 0.7867)
CALIB_TUNE_IMAGES = 500                   # seeded subset of SN23-valid for threshold selection
CALIB_ISSIA_FRAMES = 200                  # seeded, per camera
CALIB_LP_BROADCAST_CLIPS = 40             # per half, of CALIB_CLIP_FRAMES consecutive play frames
CALIB_LP_STATIC_CLIPS = 10                # per camera and half
CALIB_CLIP_FRAMES = 30
CALIB_BOOTSTRAP_DRAWS = 2000
CALIB_FOCAL_MAX_DISAGREE = 0.10           # between the two homography constraints on f
CALIB_ORACLE_MIN_JAC = 0.9                # oracle camera must fit its own annotated lines
CALIB_SPEED_IMAGES = 200
CALIB_SPEED_WARMUP = 20
CALIB_SPEED_REPEATS = 3
