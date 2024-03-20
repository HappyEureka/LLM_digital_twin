# please fill in the following parts
API_KEY = ""
API_TYPE = ""
API_BASE = ""
API_VERSION = ""
DEPLOYMENT_NAME = ""
DEPLOYMENT_NAME_BACKUP = ""

# policy constants
FIXED_POLICY = 26

# enviornment constants
ROOM_AREA = 20000 # avg room area
ROOM_HEIGHT = 3 # m

VOLUME = ROOM_AREA * ROOM_HEIGHT
DENSITY = 1.275 # density of dry air: 1.275 kg/L
#m = VOLUME * DENSITY
c_p = 1.00 # dry air specific heat: 1.00 J/(g*K)

AMBIENT_TEMP = 31

ALPHA_ = 2.2 # 2.5
BETA_ = 1/220 # 1/280