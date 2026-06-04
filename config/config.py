import os
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# ROOT_DIR = os.path.dirname(os.getcwd()) # in jupyter

LOG_DIR = os.path.join(os.path.join(ROOT_DIR, "logs"), f"{datetime.now().strftime('%Y%m')}") # script
DATAFIELDS_PATH_C = os.path.join(ROOT_DIR, "DataFields")
OPERATORS_PATH_C = os.path.join(ROOT_DIR, "Operators")
CREDENTIALS_PATH = os.path.join(ROOT_DIR, "Credentials")
CREDENTIALS_FILE_NAME = "brain_credentials_0.txt" # content format: ["username", "password"]
ERROR_401 = "401"
ERROR_429 = "429"
UNKNOWN_ERROR = "Unknown Error"
EMPTY_CONTENT_ERROR = "Empty Content Error"
SIMULATION_RETRY_TIMEOUT_ERROR = "Simulation Retry Timeout Error"
SIMULATION_REQUEST_TIMEOUT_ERROR = "Simulation Request Timeout Error"
ALREADY_SUBMITTED_ERROR = "ALREADY_SUBMITTED"
DATAFIELDS_ERROR = "DataFields Error"
OPERATOR_ERROR = "Operator Error"
VARIABLE_ERROR = "Variable Error"
PAYLOAD_ERROR = "Payload Error"
INVALID_VALUE_ERROR = "Invalid Value"
CORR_NO_RECORD_ERROR = "Correlation No Record Error"
UNEXPECTED_CHARACTER = "Unexpected character"
WARNING_ERROR = "Warning Error"
REGULAR_ERROR = "Regular Error"
PERMISSION_ERROR = "Location"
NONETYPE_GET_ERROR = "'NoneType' object has no attribute 'get'" # Invalid session or credentials (Login failed)
NONETYPE_POST_ERROR = "'NoneType' object has no attribute 'post'" # Invalid session or credentials (Login failed)
IS_CHECK_ERROR = "IS Check Error"
INVALID_ERROR = "Invalid Error"
INCORRECT_DIMENSION_ERROR = "Incorrect dimension error"
CORR_NO_RECORD_ERROR = "Correlation No Record Error"
ATTRIBUTE_ERROR = "Attribute Error"
DICT_DOESNOT_EXIST = "Dict Doesnot Exist"

NON_RETRYABLE_ERRORS = {
    PERMISSION_ERROR, NONETYPE_GET_ERROR, 
    NONETYPE_POST_ERROR, EMPTY_CONTENT_ERROR, 
    ALREADY_SUBMITTED_ERROR, DATAFIELDS_ERROR,
    OPERATOR_ERROR, VARIABLE_ERROR, UNEXPECTED_CHARACTER, 
    REGULAR_ERROR, ERROR_401, PAYLOAD_ERROR, CORR_NO_RECORD_ERROR,
    IS_CHECK_ERROR, INVALID_ERROR, INCORRECT_DIMENSION_ERROR,
    ATTRIBUTE_ERROR
}

OPERATOR_DICT = {
    "*": "×",
    "/": "÷",
    ">": "＞",
    "<": "＜",
    ":": "：",
    "?": "？",
    '"': "``",
    "'": "`",
}
REPLACE_RULES = {
    " ": "",
    "+": " + ",
    "-": " - ",
    "*": " * ",
    "/": " / ",
    "<": " < ",
    ">": " > ",
    "#": " # ",
    "=": " = ",
    "=  =": "==",
    "<  =": "<=",
    ">  =": ">=",
    "&&": " && ",
    "||": " || ",
    "?": " ? ",
    ",": ", ",
    ";": "; ",
    ":": " : ",
    " ,": ",",
    " ;": ";",
    "\n": " ",
    "  ": " ",
}

# Note that setting is for WQB API (Original Para - Abbr)
Setting_Abbr = {
    # instrumentType
    "EQUITY": "E",
    "CRYPTO": "CR", # for grand consultant
    # Neutralization
    "NONE": "N",
    "MARKET": "M",
    "INDUSTRY": "In",
    "SECTOR": "Se",
    "SUBINDUSTRY": "Su",
    # Neutralization for consultant
    "REVERSION_AND_MOMENTUM": "Ra", # RAM
    "STATISTICAL": "St",
    "CROWDING": "Cr", # CROWDING FACTORS
    "FAST": "Fa", # FAST FACTORS
    "SLOW": "Sl", # SLOW FACTORS
    "SLOW_AND_FAST": "SF", # SLOW + FAST FACTORS
    "COUNTRY": "Co",
    # Type (SUPER is only for consultant)
    "REGULAR": "R",
    "SUPER": "SA",
    # Visualization (only for consultant)
    "True": "T",
    "False": "F",
    # Region
    "USA": "USA",
    # Region for consultant
    "GLB": "GL",
    "EUR": "EU",
    "ASI": "AS",
    "CHN": "CN",
    "IND": "IND", # New since 2025-10
    "KOR": "KO",
    "TWN": "TW",
    "MEA": "MEA", # New since 2026-05
    # Universe
    "TOP3000": "3000",
    "TOP2000": "2000",
    "TOP1000": "1000",
    "TOP500": "500",
    "TOP400": "400",
    "TOP300": "300",
    "TOP200": "200",
    "TOPSP500": "SP500",
    # Universe for consultant
    "TOPDIV3000": "D3000",
    "TOP2500": "2500",
    "TOP2000U": "2000U",
    "TOP1200": "1200",
    "TOP800": "800",
    "TOP400": "400",
    "TOP100": "100",
    "ILLIQUID_MINVOL1M": "IM1M",
    "MINVOL1M": "M1M",
    "MINVOL10M": "M10M",
    "TOPCS1600": "CS1600", # New
    "TOP600": "600",
    # for super alpha only
    # selectionHandling
    "NON_ZERO": "NZ",
    "POSITIVE": "P",
    "NON_NAN": "NN",
    # componentActivation
    "IS": "IS",
    "OS": "OS"
}

Abbr_To_Full = {value: key for key, value in Setting_Abbr.items()}

REGION = {"USA", "GLB", "EUR", "ASI", "CHN", "IND", "KOR", "TWN", "MEA"} # 🚨 for  consultant gold

UNIVERSE = {
    "USA": {"TOP3000", "TOP2000", "TOP1000", "TOP500", "TOP200", "TOPSP500", "ILLIQUID_MINVOL1M"},
    "GLB": {"TOP3000", "MINVOL1M", "MINVOL10M", "TOPDIV3000"},
    "EUR": {"TOP2500", "TOP1200", "TOP800", "TOP400", "ILLIQUID_MINVOL1M", "TOPCS1600"},
    "ASI": {"TOP500", "MINVOL1M", "MINVOL10M", "ILLIQUID_MINVOL1M"},
    "CHN": {"TOP2000U"},
    "IND": {"TOP500"},
    "KOR": {"TOP600"},
    "TWN": {"TOP500", "TOP100"},
    "MEA": {"TOP400", "TOP300"} # New since 2026-05
} # 🚨 for consultant gold

DELAY = {
    "USA": {0, 1},
    "GLB": {1},
    "EUR": {0, 1},
    "ASI": {1},
    "CHN": {0, 1},
    "IND": {1},
    "KOR": {1},
    "TWN": {1},
    "MEA": {1} # New since 2026-05
} # 🚨 for consultant gold

DEFAULT_SETTINGS = {
    "unitHandling": "VERIFY",
    "language": "FASTEXPR",
    "visualization": False
} # default settings for all regions, can be updated by user input

MAX_TRADE = {
    "USA": {"ON", "OFF"}, # ✅
    "GLB": {"ON", "OFF"}, # ✅
    "EUR": {"ON", "OFF"}, # ✅
    "ASI": {"ON", "OFF"}, # ✅
    "CHN": {"ON", "OFF"}, # ✅
    "IND": {"ON", "OFF"}, # ✅
    "KOR": {"ON", "OFF"}, # ✅
    "TWN": {"ON", "OFF"}, # ✅
    "MEA": {"ON", "OFF"} # New since 2026-05 ✅
} # 🚨 for consultant gold

MAX_POSITION = {
    "USA": {"ON", "OFF"}, # ✅
    "GLB": {"OFF"}, # ✅
    "EUR": {"ON", "OFF"}, # ✅
    "ASI": {"ON", "OFF"}, # ✅
    "CHN": {"OFF"}, # ✅
    "IND": {"OFF"}, # ✅
    "KOR": {"OFF"}, # ✅
    "TWN": {"OFF"}, # ✅
    "MEA": {"ON", "OFF"}, # New since 2026-05 ✅
} # 🚨 Complete Max Position Dict (for consultant gold)

FULL_NEUTRALIZATION = {
    "MARKET", "SECTOR", "INDUSTRY", 
    "SUBINDUSTRY", "REVERSION_AND_MOMENTUM", 
    "STATISTICAL", "CROWDING", "FAST", "SLOW", 
    "SLOW_AND_FAST", "COUNTRY"
} # 🚨 for consultant gold

NEUTRALIZATION_DICT = {
    "USA": FULL_NEUTRALIZATION - {"COUNTRY"},
    "GLB": FULL_NEUTRALIZATION,
    "EUR": FULL_NEUTRALIZATION,
    "ASI": FULL_NEUTRALIZATION,
    "CHN": FULL_NEUTRALIZATION - {"COUNTRY", "STATISTICAL"},
    "IND": FULL_NEUTRALIZATION - {"COUNTRY", "STATISTICAL"},
    "KOR": FULL_NEUTRALIZATION - {"COUNTRY", "STATISTICAL"},
    "TWN": FULL_NEUTRALIZATION - {"COUNTRY", "STATISTICAL"},
    "MEA": FULL_NEUTRALIZATION - {"REVERSION_AND_MOMENTUM", "STATISTICAL", "CROWDING", "FAST", "SLOW", "SLOW_AND_FAST"} # New since 2026-05
} # 🚨 for consultant gold

SELECTIONHANDLING = {"NON_ZERO", "POSITIVE", "NON_NAN"} # 🚨 for super alpha only
COMPONENTACTIVATION = {"IS", "OS"} # 🚨 for super alpha only

MUST_INCLUDE_SETTINGS_KEYS = [
    "instrumentType",
    "region",
    "universe",
    "delay",
    "decay",
    "neutralization",
    "truncation",
    "pasteurization",
    "nanHandling",
    "testPeriod",
    "maxTrade",
    "maxPosition"
] # must be included in the main dict of alpha_settings for simplicity, no sub dict for settings

IS_Check_Name_Dict = {  # lowercase
    "LOW_SHARPE": "Sharpe",
    "LOW_FITNESS": "Fitness",
    "LOW_TURNOVER": "Turnover",
    "HIGH_TURNOVER": "Turnover",
    "CONCENTRATED_WEIGHT": "Concentrated Weight",
    "LOW_SUB_UNIVERSE_SHARPE": "Sub-universe Sharpe",
    "SELF_CORRELATION": "Self-correlation",
    "MATCHES_COMPETITION": "Matches Competition",
    "UNITS": "UNITS",
    # consultant only
    "DATA_DIVERSITY": "Data Diversity",
    "PROD_CORRELATION": "Prod Correlation",
    "REGULAR_SUBMISSION": "Regular Submission",
    "IS_LADDER_SHARPE": "IS Ladder Sharpe",
    "MATCHES_PYRAMID": "Matches Pyramid",
    "MATCHES_THEMES": "Matches Themes",
    "LOW_2Y_SHARPE": "2 Year Sharpe",
    # for super
    "SUPER_SUBMISSION": "Super Submission",
    "COMBO_DESCRIPTION_LENGTH": "Combo Description Length",
    "SELECTION_DESCRIPTION_LENGTH": "Selection Description Length"
}