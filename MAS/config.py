import os


#Setup
FIRST_START = 1
ROOT = os.path.dirname(__file__)

#Model path
DLCDETECTOR = os.environ.get('MAS_DLC_CONFIG', os.path.join(ROOT, 'Models', 'PRT-Benj-2024-01-30', 'config.yaml'))
NESTDETECTOR = os.environ.get('MAS_NEST_MODEL', os.path.join(ROOT, 'Models', 'Nest10'))

#Misc
DAMSIZECM = 8.5
MIN_AREA_THRESHOLD = 250
NEST_BORDER_THRESHOLD = 10
DLC_THRESHOLD = 0.7