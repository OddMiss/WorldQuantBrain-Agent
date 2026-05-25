import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import ast
import json
import os
import re
import sys
import time
from decimal import Decimal, localcontext, getcontext, MAX_EMAX, MIN_EMIN
from typing import Any, Dict, Optional, Tuple
import requests
import logging
from requests.auth import HTTPBasicAuth
from config.config import *
from utils.logger import setup_logger, FlagLogger

# logger = setup_logger(LOG_DIR, "alpha_simulator")

def _retry_after(FUNCTION, *args, **kwargs):
    """
    - example usage:
        _retry_after(
            FUNCTION=Single_Alpha_Simulation,
            alpha_id=alpha_id, # *args
            account_no=account_no
        )
    - MAX_RETRIES = kwargs.pop("MAX_RETRIES", 3) # for RETRY_EXCEPT only
    - R_account_no = kwargs.get("account_no", "N") # get from FUNCTION args or kwargs
    - function_name = kwargs.pop("F_NAME", "") # for RETRY_EXCEPT only
    """
    MAX_RETRIES = kwargs.pop("MAX_RETRIES", 3)
    R_account_no = kwargs.get("account_no", "N")
    function_name = kwargs.pop("F_NAME", "")
    if not function_name: function_name = FUNCTION.__name__
    for attempt in range(MAX_RETRIES):
        time.sleep(3)
        try:
            fail, value = FUNCTION(*args, **kwargs)
            if not fail: return False, value
            raise Exception(value) # Raise using the detailed message
        except Exception as e:
            e_msg = str(e)
            flag = "❗❗❗❗❗❗❗❗❗❗" if attempt >= 1 else ""
            logger.error(f"RETRY EXCEPT-{function_name} ({R_account_no})", f"Attempt {attempt + 1}{flag} failed: due to {e_msg}")
            
            # Check the general classifier flag to stop retrying
            if fail in NON_RETRYABLE_ERRORS: return True, e_msg
            if attempt == MAX_RETRIES - 1:
                logger.error(f"RETRY EXCEPT-{function_name} ({R_account_no})", f"All attempts failed.")
                return True, e_msg
            logger.info(f"RETRY EXCEPT-{function_name} ({R_account_no})", f"Retrying after 5 seconds... (Attempt {attempt + 2} of {MAX_RETRIES})")

def _dict_to_jsonFile(Dict, filename):
    # Attention that add `ensure_ascii=False` for encoding Chinese correctly
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(Dict, json_file, ensure_ascii=False, indent=4)

def _jsonFile_to_dict(filename, account_no="N"):
    try:
        # JSON file to Dict
        with open(filename, "r", encoding="utf-8") as json_file: Dict = json.load(json_file)
        return Dict
    except Exception as e:
        logger.error(f"Json to Dict {account_no}", f"Invalid json file format with {e}")
        return {}

def _get_pyramid_multipliers(ROOTPATH, SESS=None, Online=False, Save_to_Local=False, account_no="N"):
    time.sleep(3)
    if Online:
        api_url = "https://api.worldquantbrain.com/users/self/activities/pyramid-multipliers"
        alpha_resp = SESS.get(api_url)
        if alpha_resp.status_code == 200:
            pyramid_data = alpha_resp.json()
            pyramid_dict = {}
            for item in pyramid_data["pyramids"]:
                ID = (item["category"]["name"], item["region"], item["delay"])
                pyramid_dict[str(ID)] = item["multiplier"]
            # sort the dict by value (multiplier) in descending order (largest multiplier first)
            pyramid_dict = dict(sorted(pyramid_dict.items(), key=lambda x: x[1], reverse=True))
            logger.info(f"Get Pyramid Multipliers ({account_no})", f"✅ Pyramid multipliers fetched successfully from API.")
            if Save_to_Local: _dict_to_jsonFile(pyramid_dict, os.path.join(ROOTPATH, "Pyramid-Multipliers.json"))
            return False, pyramid_dict
        else:
            Status_Code = alpha_resp.status_code
            if Status_Code == 401: return ERROR_401, f"❌ Unauthorized (401): Failed to authenticate. Please check your credentials and session. Response: {alpha_resp.text}"
            elif Status_Code == 429: return ERROR_429, f"❌ Rate Limited (429): Too many requests. Please try again later. Response: {alpha_resp.text}"
            else:
                logger.error(f"Get Pyramid Multipliers ({account_no})", f"❌ Error: {Status_Code}")
                return UNKNOWN_ERROR, f"❌ Unknown Error: {Status_Code}"
    else: 
        pyramid_dict = _jsonFile_to_dict(os.path.join(ROOTPATH, "Pyramid-Multipliers.json"), account_no=account_no)
        logger.info(f"Get Pyramid Multipliers ({account_no})", f"✅ Pyramid multipliers loaded successfully from local file.")
        return False, pyramid_dict

def _get_datasetid_suffix(ROOTPATH, account_no="N"):
    DatasetID_Suffix_Dict = _jsonFile_to_dict(os.path.join(ROOTPATH, "Dataset-Suffix-Category.json"), account_no=account_no)
    logger.info(f"Get DatasetID Suffix ({account_no})", f"✅ DatasetID suffix-category dict loaded successfully from local file.")
    return DatasetID_Suffix_Dict

def _get_datafieldid_datasetid_suffix(ROOTPATH, account_no="N"):
    Datafield_Dataset_Dict = _jsonFile_to_dict(os.path.join(ROOTPATH, "Datafield-Dataset-Category.json"), account_no=account_no)
    logger.info(f"Get Datafield-Dataset ({account_no})", f"✅ Datafield-Dataset dict loaded successfully from local file.")
    return Datafield_Dataset_Dict

def _get_operators(ROOTPATH, SESS=None, Online=False, Download_2_Local=False, account_no="N"):
    """
    - Fetch available operators from WorldQuant Brain.
    - The output key is the the operator name, and the value is the operator details. (The agent can directly call the operator by name without category)
    The category information is still included in the operator details, so the agent can also choose to call operators by category if needed.
    """
    if Online:
        response = SESS.get('https://api.worldquantbrain.com/operators')
        if response.status_code != 200:
            logger.error(f"Get Operators ({account_no})", f"❌ Failed to get operators: {response.text} (Status Code: {response.status_code})")
            return True, f"❌ Failed to get operators: {response.text}"
        data = response.json()
        # print(type(data))  # List
        # The operators endpoint might return a direct array instead of an object with 'items' or 'results'
        if isinstance(data, list):
            # Turn to dict with category
            Operators_Category_Dict = {param['category']: {} for param in data if 'category' in param}
            Operators_Dict = {} # The key is operator name, and the value is operator details
            for operator in data:
                Operators_Category_Dict[operator["category"]] |= {operator["name"]: operator}
                Operators_Dict[operator["name"]] = operator
            for category in Operators_Category_Dict:
                length = len(Operators_Category_Dict[category])
                temp_operator_list = list(Operators_Category_Dict[category].keys())
                logger.info(f"Get Operators ({account_no})", f"✅ Category: '{category}' with {length} operators {temp_operator_list} finished.")
            if Download_2_Local:
                Operator_Path = os.path.join(ROOTPATH, "Operators-Agent.json")
                os.makedirs(os.path.dirname(Operator_Path), exist_ok=True)
                _dict_to_jsonFile(Operators_Dict, Operator_Path)
                logger.info(f"Get Operators ({account_no})", f"✅ Operators Saved to {Operator_Path}")
            return False, Operators_Dict
        elif 'results' in data: return True, data['results']
        else: 
            logger.error(f"Get Operators ({account_no})", f"❌ Unexpected operators response format. Response: {data}")
            return True, f"❌ Unexpected operators response format. Response: {data}"
    else:
        Operator_Path = os.path.join(ROOTPATH, "Operators-Agent.json")
        Operators_Dict = _jsonFile_to_dict(Operator_Path, account_no=account_no)
        logger.info(f"Get Operators ({account_no})", f"✅ Operators loaded successfully from local file.")
        return False, Operators_Dict

global FIELD_SET_SUFFIX, SET_SUFFIX, OPERATOR_INFO_DICT, PYRAMID_MULTIPLIERS
def initialize_global_variables(account_no="N"):
    global FIELD_SET_SUFFIX, SET_SUFFIX, OPERATOR_INFO_DICT, PYRAMID_MULTIPLIERS
    FIELD_SET_SUFFIX = _get_datafieldid_datasetid_suffix(DATAFIELDS_PATH_C, account_no=account_no)
    SET_SUFFIX = _get_datasetid_suffix(DATAFIELDS_PATH_C, account_no=account_no)
    fail, OPERATOR_INFO_DICT = _retry_after(FUNCTION=_get_operators, ROOTPATH=OPERATORS_PATH_C, Online=False, account_no=account_no)
    if fail:
        logger.error(f"Initialize Global Variables ({account_no})", f"❌ Failed to get operators with error: {OPERATOR_INFO_DICT}")
        return True, f"❌ Failed to get operators with error: {OPERATOR_INFO_DICT}"
    fail, PYRAMID_MULTIPLIERS = _retry_after(FUNCTION=_get_pyramid_multipliers, ROOTPATH=DATAFIELDS_PATH_C, account_no=account_no)
    if fail:
        logger.error(f"Initialize Global Variables ({account_no})", f"❌ Failed to get pyramid multipliers with error: {PYRAMID_MULTIPLIERS}")
        return True, f"❌ Failed to get pyramid multipliers with error: {PYRAMID_MULTIPLIERS}"
    logger.info(f"Initialize Global Variables ({account_no})", f"✅ Global variables initialized successfully.")
    return False, "✅ Global variables initialized successfully."

def _build_simulation_payload(alpha_settings, regular_formula, account_no="N"):
    """
    - Builds a simulation payload based on the provided alpha settings and regular formula.
    - alpha_settings: dict containing "type" (REGULAR or SUPER) and "settings" (dict of settings).
    There is no sub dict for settings, all keys are expected to be in the main dict for simplicity. 
    settings keys include instrumentType, region, universe, delay, decay, neutralization, truncation, 
    pasteurization, nanHandling, testPeriod, maxTrade, visualization. 
    default values will be used if not provided.
    """
    if not isinstance(alpha_settings, dict):
        logger.error(f"Build Simulation Payload ({account_no})", "❌ alpha_settings must be a dict.")
        return PAYLOAD_ERROR, f"❌ alpha_settings must be a dict."

    alpha_type = alpha_settings.get("type", "REGULAR")
    settings = {}
    
    # Check if all required keys are present in alpha_settings
    missing_keys = [key for key in MUST_INCLUDE_SETTINGS_KEYS if key not in alpha_settings]
    if missing_keys:
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Missing required settings keys: {', '.join(missing_keys)}")
        return PAYLOAD_ERROR, f"❌ Missing required settings keys: {', '.join(missing_keys)}"

    # Check region
    region = alpha_settings.get("region")
    if region not in REGION:
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid region: {region}, must be one of {REGION}.")
        return PAYLOAD_ERROR, f"❌ Invalid region: {region}, must be one of {REGION}."

    # Check universe
    universe = alpha_settings.get("universe")
    if universe not in UNIVERSE.get(region, set()):
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid universe: {universe} for region: {region}, must be one of {UNIVERSE.get(region, set())}.")
        return PAYLOAD_ERROR, f"❌ Invalid universe: {universe} for region: {region}, must be one of {UNIVERSE.get(region, set())}."

    # Check delay (must be integer, and valid for the region)
    delay = alpha_settings.get("delay")
    if not isinstance(delay, int):
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid delay: {delay}, must be an integer.")
        return PAYLOAD_ERROR, f"❌ Invalid delay: {delay}, must be an integer."
    if delay not in DELAY.get(region, set()):
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid delay: {delay} for region: {region}, must be one of {DELAY.get(region, set())}.")
        return PAYLOAD_ERROR, f"❌ Invalid delay: {delay} for region: {region}, must be one of {DELAY.get(region, set())}."
    
    # Check decay (between 0 and 512, suggest to be integer for better performance, but float is also acceptable)
    decay = alpha_settings.get("decay")
    if not isinstance(decay, (int, float)):
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid decay: {decay}, must be a number.")
        return PAYLOAD_ERROR, f"❌ Invalid decay: {decay}, must be a number."
    if decay < 0 or decay > 512:
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid decay: {decay}, must be a number between 0 and 512, inclusive.")
        return PAYLOAD_ERROR, f"❌ Invalid decay: {decay}, must be a number between 0 and 512, inclusive."

    # Check neutralization
    neutralization = alpha_settings.get("neutralization")
    if neutralization not in NEUTRALIZATION_DICT.get(region, set()):
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid neutralization: {neutralization} for region: {region}, must be one of {NEUTRALIZATION_DICT.get(region, set())}.")
        return PAYLOAD_ERROR, f"❌ Invalid neutralization: {neutralization} for region: {region}, must be one of {NEUTRALIZATION_DICT.get(region, set())}."
    
    # Check truncation (between 0 and 1, inclusive, float only, suggest keep 2 decimal places for better performance, but more decimal places are also acceptable)
    truncation = alpha_settings.get("truncation")
    if not isinstance(truncation, (int, float)):
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid truncation: {truncation}, must be a number.")
        return PAYLOAD_ERROR, f"❌ Invalid truncation: {truncation}, must be a number."
    if truncation < 0 or truncation > 1:
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid truncation: {truncation}, must be a number between 0 and 1, inclusive.")
        return PAYLOAD_ERROR, f"❌ Invalid truncation: {truncation}, must be a number between 0 and 1, inclusive."
    
    # Check pasteurization (either ON or OFF, case-sensitive)
    pasteurization = alpha_settings.get("pasteurization")
    if pasteurization not in {"ON", "OFF"}:
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid pasteurization: {pasteurization}, must be either 'ON' or 'OFF'.")
        return PAYLOAD_ERROR, f"❌ Invalid pasteurization: {pasteurization}, must be either 'ON' or 'OFF'."

    # Check nanHandling (either ON or OFF, case-sensitive)
    nan_handling = alpha_settings.get("nanHandling")
    if nan_handling not in {"ON", "OFF"}:
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid nanHandling: {nan_handling}, must be either 'ON' or 'OFF'.")
        return PAYLOAD_ERROR, f"❌ Invalid nanHandling: {nan_handling}, must be either 'ON' or 'OFF'."

    # Check maxTrade
    max_trade = alpha_settings.get("maxTrade")
    if max_trade not in MAX_TRADE.get(region, set()):
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid maxTrade: {max_trade} for region: {region}, must be one of {MAX_TRADE.get(region, set())}.")
        return PAYLOAD_ERROR, f"❌ Invalid maxTrade: {max_trade} for region: {region}, must be one of {MAX_TRADE.get(region, set())}."

    # Check maxPosition (either ON or OFF, case-sensitive)
    max_position = alpha_settings.get("maxPosition")
    if max_position not in MAX_POSITION.get(region, set()):
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid maxPosition: {max_position}, must be either 'ON' or 'OFF'.")
        return PAYLOAD_ERROR, f"❌ Invalid maxPosition: {max_position}, must be either 'ON' or 'OFF'."

    # Max Position and Max Trade cannot both be set to On simultaneously
    if max_position == "ON" and max_trade == "ON":
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid combination: both maxPosition and maxTrade are set to 'ON'.")
        return PAYLOAD_ERROR, f"❌ Invalid combination: both maxPosition and maxTrade are set to 'ON'."

    # Check testPeriod
    # Format 1: with year and without month, like "P1Y"
    # Format 2: with year and month, like "P1Y2M"
    # Format 3: with month only, like "P2M"
    test_period = alpha_settings.get("testPeriod")
    if not isinstance(test_period, str):
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid testPeriod: {test_period}, must be a string in format like 'P1Y', 'P1Y2M', or 'P2M'.")
        return PAYLOAD_ERROR, f"❌ Invalid testPeriod: {test_period}, must be a string in format like 'P1Y', 'P1Y2M', or 'P2M'."
    if not re.match(r'^P(\d+Y)?(\d+M)?$', test_period):
        logger.error(f"Build Simulation Payload ({account_no})", f"❌ Invalid testPeriod format: {test_period}, must be in format like 'P1Y', 'P1Y2M', or 'P2M'.")
        return PAYLOAD_ERROR, f"❌ Invalid testPeriod format: {test_period}, must be in format like 'P1Y', 'P1Y2M', or 'P2M'."

    for key in MUST_INCLUDE_SETTINGS_KEYS:
        settings[key] = alpha_settings[key]
    
    # Fill in default settings if not provided
    for key, value in DEFAULT_SETTINGS.items():
        settings.setdefault(key, value)
    
    payload = {"type": alpha_type, "settings": settings}
    
    # the payload for simulation
    payload["regular"] = regular_formula

    return False, payload

def _format_regular_formula(regular_formula: str) -> str:
    for old, new in REPLACE_RULES.items():
        regular_formula = regular_formula.replace(old, new)
    return regular_formula.strip()

# def _replace_regular_operators(regular_formula: str) -> str:
#     for old, new in OPERATOR_DICT.items():
#         regular_formula = regular_formula.replace(old, new)
#     return regular_formula

def _parse_alpha_expression(alpha_expression: str) -> Dict[str, Any]:
    """
    From: https://support.worldquantbrain.com/hc/en-us/community/posts/30870358077463
    
    Parses an alpha expression and extracts operators and data fields.
    This function processes a given alpha expression by converting ternary expressions,
    fixing indentation errors, and parsing it into an abstract syntax tree (AST). It then
    traverses the AST to extract operators and data fields, while filtering out defined
    variables, NaN values, and Python built-in functions and keywords.
    Args:
        alpha_expression (str): The alpha expression to be parsed.
    Returns:
        dict: A dictionary containing two lists:
        - 'operators': A list of unique operators (function and method names) found in the expression.
        - 'data_fields': A list of unique data fields (variable names) found in the expression,
        excluding defined variables and NaN values.
    """
    # Preprocessing: Remove // and # comments
    alpha_expression = re.sub(r'(?m)^\s*(//|#).*\n?', '', alpha_expression)
    # Handle ternary expressions
    alpha_expression = alpha_expression.replace('?', ' if ').replace(':', ' else ')
    # Resolve conflicts with Python built-in logical expressions
    alpha_expression = re.sub(r'\band\b', 'and_', alpha_expression)  # Replace standalone 'and'
    alpha_expression = re.sub(r'\band\(', 'and_(', alpha_expression)  # Replace 'and('
    alpha_expression = re.sub(r'\bor\b', 'or_', alpha_expression)  # Replace standalone 'or'
    alpha_expression = re.sub(r'\bor\(', 'or_(', alpha_expression)  # Replace 'or('
    # Handle logical expressions
    alpha_expression = alpha_expression.replace('!', ' not ').replace('&&', ' and ')
    # Handle expressions like range="0.1,1,0.1"
    alpha_expression = re.sub(r'"([^"]*)"', "0", alpha_expression)
    # Fix indentation errors
    alpha_expression = "\n".join(line.strip() for line in alpha_expression.splitlines())
    # Parse expression into Abstract Syntax Tree (AST)
    tree = ast.parse(alpha_expression)
    # Extract operators and data fields
    operators = set()
    data_fields = set()
    # Variables defined during extraction
    defined_variables = set()
    # Traverse the AST
    for node in ast.walk(tree):
        # Extract variable names from assignment statements
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_variables.add(target.id)  # Record defined variable names
        # Extract function calls (operators)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                operators.add(node.func.id)  # Function name
            elif isinstance(node.func, ast.Attribute):
                operators.add(node.func.attr)  # Method name
        # Extract variable names (data fields)
        if isinstance(node, ast.Name):
            data_fields.add(node.id)  # Variable name
        # Extract ternary conditional expressions
        if isinstance(node, ast.IfExp):
            # Extract condition part
            if isinstance(node.test, ast.Compare):
                for comparator in node.test.comparators:
                    if isinstance(comparator, ast.Name):
                        data_fields.add(comparator.id)
            # Extract if part
            if isinstance(node.body, ast.Name):
                data_fields.add(node.body.id)
            # Extract else part
            if isinstance(node.orelse, ast.Name):
                data_fields.add(node.orelse.id)
    # Filter out variables defined during the process
    data_fields = data_fields - defined_variables
    # Filter out special variables: nan
    data_fields = data_fields - set(['nan', 'true', 'false'])
    # Filter out Python built-in functions and keywords
    builtin_functions = set(dir(__builtins__))  # Python built-in functions
    operators = operators - builtin_functions  # Remove built-in functions
    data_fields = data_fields - builtin_functions - operators  # Remove built-in functions
    return {'operators': list(operators), "data_fields": list(data_fields)}

def _float_calulator(*float_num):
    """
    - Float calculator to avoid overflow.
    - Input can be float or string. Returns the sum as a string to prevent overflow.
    """
    if not float_num: return "0"
    ctx = getcontext().copy()
    ctx.Emax = MAX_EMAX
    ctx.Emin = MIN_EMIN
    ctx.prec = 1000
    with localcontext(ctx):
        total = Decimal(0)
        for num in float_num:
            total += Decimal(str(num)) if isinstance(num, float) else Decimal(num)
    return float(total)

def _combined_multiplier(datasetName_series, pyramid_multipliers_dict, region, delay):
    """
    - In `WQB-Alpha.py`
    - final multiplier = sum of all multipliers – number of multipliers + 1
    """
    multiplier_list = []
    for datasetname in datasetName_series:
        key = str((datasetname, region, int(delay)))
        multiplier = pyramid_multipliers_dict.get(key, 1)
        multiplier_list.append(multiplier)
    Result = _float_calulator(_float_calulator(*multiplier_list), -len(multiplier_list), 1)
    return Result

def _check_regular_format(
    regular, field_set_suffix, set_suffix, 
    pyramid_multipliers, region, delay, 
    universe, account_no="N"
):
    """
    - Check the format of a regular formula and extract its components for single settings.
    - regular -> datafields (`Parse_Alpha`) -> Datasetname (`field_set_suffix`)
    -> Suffix (`set_suffix`) -> params (`Suffix2ComParams`) -> coms (`Generate_Combination_Dict_C`)
    - regular: must be a string
    - field_set_suffix: dict from `Get_DatafieldID_DatasetID_Suffix`
    - set_suffix: dict from `Get_DatasetID_Suffix`
    - pyramid_multipliers: dict from `Get_Pyramid_Multipliers`
    - region and delay: for multiplier calculation
    """
    try:
        parsed = _parse_alpha_expression(regular)
        Datafields = parsed["data_fields"] # get all datafields from the regular
        Operators = parsed["operators"] # get all operators from the regular
    except (SyntaxError, ValueError) as exc:
        return REGULAR_ERROR, f"❌ {REGULAR_ERROR}: invalid regular syntax ({exc})"
    except Exception as exc:
        return REGULAR_ERROR, f"❌ {REGULAR_ERROR}: unable to parse regular formula ({type(exc).__name__})."
    if not parsed.get("data_fields"):
        return REGULAR_ERROR, f"❌ {REGULAR_ERROR}: no data fields detected."
    logger.info(f"Check Regular Format ({account_no})", f"✅ the regular formula is parsed successfully.")
    logger.info(f"Check Regular Format ({account_no})", f"Datafields: {Datafields}")
    logger.info(f"Check Regular Format ({account_no})", f"Operators: {Operators}")
    # Check if all datafields exist in the field_set_suffix dict
    Suffix_List = []
    Dataset_List = []
    Dataset_Name_list = [] # or category name
    ERROR_COUNT = 0
    return_value = ""
    for datafield in Datafields:
        datafield_dict = field_set_suffix.get(datafield, {})
        if datafield_dict: dataset_suffix = datafield_dict["dataset_suffix"] # get dataset_suffix by datafield
        else: 
            error_str = f"❌ {datafield} doesn't exist."
            logger.error(f"Check Regular Format ({account_no})", error_str)
            return_value += error_str
            ERROR_COUNT += 1
            continue
        dataset_id = set([item.split("-")[0] for item in dataset_suffix]) # get dataset id
        suffix = set(["-".join(item.split("-")[1:]) for item in dataset_suffix]) # get suffix
        Dataset_List.append(dataset_id)
        Suffix_List.append(suffix)
    if ERROR_COUNT > 0: return REGULAR_ERROR, return_value
    Datasetid_Set = set.union(*Dataset_List) # union all dataset id (to calculate multiplier)
    for datasetid in Datasetid_Set:
        dataset_name = set_suffix[datasetid]["categoty_name"]
        Dataset_Name_list.append({dataset_name})
    Datasetname_Series = set.union(*Dataset_Name_list)
    logger.info(f"Check Regular Format ({account_no})", f"Datasetname Series: {Datasetname_Series}")
    Suffix_Series = set.intersection(*Suffix_List) # intersect all suffix (to get com params)
    # The format of items in Suffix_Series are like "E-AS-1-IM1M" (abbreviation format)
    # "E" (EQUITY), region "AS" (Asia), delay "1", and universe "IM1M" (ILLIQUID_MINVOL1M).
    # logger.info(f"Check Regular Format ({account_no})", f"Suffix Series: {Suffix_Series}")
    
    # Convert the abbreviation to full name, and check if the region in suffix matches the input region
    # The input region and universe are not in abbreviation format.
    Input_Region_Delay_Universe = f"{region}-{delay}-{universe}"
    Region_Delay_Universe_List = []
    ERROR_COUNT = 0
    return_value = ""
    for suffix in Suffix_Series:
        parts = suffix.split("-")
        if len(parts) < 4:
            error_str = f"Invalid suffix format: '{suffix}'. Expected format like 'E-AS-1-IM1M'."
            logger.error(f"Check Regular Format ({account_no})", error_str)
            ERROR_COUNT += 1
            return_value += error_str
            continue
        region_abbr = parts[1]
        delay = parts[2]
        universe_abbr = parts[3]
        region_full = Abbr_To_Full.get(region_abbr, None)
        universe_full = Abbr_To_Full.get(universe_abbr, None)
        Region_Delay_Universe_List.append(f"{region_full}-{delay}-{universe_full}")
    logger.info(f"Check Regular Format ({account_no})", f"Region-Delay-Universe combinations: {Region_Delay_Universe_List}")
    if ERROR_COUNT > 0: return REGULAR_ERROR, return_value
    if Input_Region_Delay_Universe not in Region_Delay_Universe_List:
        error_str = (
            f"❌ Input region, delay, universe '{Input_Region_Delay_Universe}' does not match any "
            f"expected combinations for this regular. Combinations: {Region_Delay_Universe_List}."
        )
        logger.error(f"Check Regular Format ({account_no})", error_str)
        return REGULAR_ERROR, error_str
    logger.info(f"Check Regular Format ({account_no})", f"✅ Input region, delay, universe '{Input_Region_Delay_Universe}' matches the expected combinations.")

    # Check if the operators in the regular are valid (either in Python or in WQB functions)
    ERROR_COUNT = 0
    return_value = ""
    for operator in Operators:
        if operator not in OPERATOR_INFO_DICT and operator not in dir(__builtins__):
            error_str = f"❌ Operator '{operator}' is not recognized."
            logger.error(f"Check Regular Format ({account_no})", error_str)
            ERROR_COUNT += 1
            return_value += error_str
    if ERROR_COUNT > 0: return REGULAR_ERROR, return_value

    # Get combined multiplier
    multiplier = _combined_multiplier(Datasetname_Series, pyramid_multipliers, region, delay)
    logger.info(f"Check Regular Format ({account_no})", f"Combined Multiplier: {multiplier}")
    output_dict = {
        "Datasetname_Series": Datasetname_Series,
        "Suffix_Series": Suffix_Series,
        "Combined_Multiplier": multiplier
    }
    return False, output_dict

def _validate_regular_formula(
    regular_formula: Optional[str], 
    region, delay, universe, 
    account_no: str="N"
) -> Tuple[bool, str]:
    if not isinstance(regular_formula, str) or not regular_formula.strip():
        return REGULAR_ERROR, "regular is required for REGULAR alphas."
    formatted_formula = _format_regular_formula(regular_formula)
    fail, result = _check_regular_format(
        formatted_formula, 
        field_set_suffix=FIELD_SET_SUFFIX, 
        set_suffix=SET_SUFFIX, 
        pyramid_multipliers=PYRAMID_MULTIPLIERS, 
        region=region, 
        delay=delay,
        universe=universe,
        account_no=account_no
    )
    if fail: return REGULAR_ERROR, result
    result.update({"formatted_formula": formatted_formula})
    return False, result

# Version 1: Login and Session Management (without local session loading)

# def _save_session_to_file(session, account_no):
#     with open(os.path.join(CREDENTIALS_PATH, f"SESS_{account_no}.json"), "w", encoding="utf-8") as f:
#         json.dump({"session": session.cookies.get_dict()}, f)
#     logger.info(f"Session Update ({account_no})", "Session saved to local file.")

# def Login_to_WQB(
#     account_no: str = "N",
#     Return_Permission: bool = False,
# ):
#     credentials_file_path = os.path.join(CREDENTIALS_PATH, CREDENTIALS_FILE_NAME)
#     with open(os.path.expanduser(credentials_file_path), encoding="utf-8") as f:
#         credentials = json.load(f)
#     if not isinstance(credentials, (list, tuple)) or len(credentials) < 2:
#         raise ValueError("Credentials file must contain [username, password]")
#     username, password = credentials[0], credentials[1]
#     sess = requests.Session()
#     sess.auth = HTTPBasicAuth(username, password)
#     response = sess.post("https://api.worldquantbrain.com/authentication")
#     status_code = response.status_code
#     if str(status_code) == ERROR_401:
#         logger.error(f"Log In ({account_no})", "❌❌❌Invalid credentials. Please check your credentials file❌❌❌.")
#         return None
#     logger.info(f"Log In ({account_no})", str(status_code))
#     logger.info(f"Log In ({account_no})", str(response.json()))
#     if Return_Permission:
#         permission_list = response.json().get("permissions", [])
#         permission_content = str(tuple(permission_list))
#         return sess, permission_content

#     # save session as a local file
#     _save_session_to_file(sess, account_no)
#     return sess

# def Check_Session_Timeout(
#     SESS,
#     threshold: int = 60,
#     account_no: str = "N"
# ):
#     if SESS is None:
#         logger.error(f"Check Session Timeout ({account_no})", "Session is None...")
#         return None
#     authentication_url = "https://api.worldquantbrain.com/authentication"
#     result = SESS.get(authentication_url)
#     if result.status_code == 200:
#         result_dict = result.json()
#         time_out = result_dict.get("token", {}).get("expiry", {})
#         if time_out:
#             time_out = time_out / 60
#             logger.info(
#                 f"Check Session Timeout ({account_no})",
#                 f"Time left for session expiry: {round(time_out, 3)} minutes"
#             )
#             if time_out < threshold:
#                 logger.info(f"Check Session Timeout ({account_no})", "re-logging...")
#                 SESS =  Login_to_WQB(account_no=account_no)
#                 return SESS
#             return SESS
#         logger.info(f"Check Session Timeout ({account_no})", "Session expired, re-authenticating...")
#     else:
#         logger.info(
#             f"Check Session Timeout ({account_no})",
#             f"Failed to check session timeout, status code: {result.status_code}. Re-authenticating..."
#         )
#         SESS = Login_to_WQB(account_no=account_no)
#         return SESS

# Version 2: Login and Session Management (with local session loading)
def _session_file_path(account_no):
    """Get the file path for storing session based on account number."""
    return os.path.join(CREDENTIALS_PATH, f"SESS_{account_no}.json")

def _save_session_to_file(session, account_no):
    """Save the session cookies to a local file for future use."""
    with open(_session_file_path(account_no), "w", encoding="utf-8") as f:
        json.dump({"session": session.cookies.get_dict()}, f)
    logger.info(f"Save Session ({account_no})", "Session saved to local file.")

def _load_session_from_file(account_no):
    """Load the session cookies from a local file if it exists and return a requests.Session object."""
    session_file = _session_file_path(account_no)
    if not os.path.exists(session_file): return None

    try:
        # Load the session data from the file
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract cookies and create a session
        cookies = data.get("session", {})
        if not isinstance(cookies, dict) or not cookies: return None

        # Create a new session and update it with the loaded cookies
        sess = requests.Session()
        sess.cookies.update(cookies)
        logger.info(f"Load Session ({account_no})", "Local session loaded.")
        return sess
    except Exception as e:
        logger.error(f"Load Session ({account_no})", f"Failed to load local session: {e}")
        return None

def Login_to_WQB(
    account_no: str = "N",
    Use_Local_Session: bool = True,
):
    """
    - Login to WQB with optional local session loading.
    - If Use_Local_Session is True, it will first attempt to load a local session from a file. 
    - If the local session is valid, it will be used for authentication. 
    - If the local session is invalid or does not exist, it will proceed with 
    normal login using credentials and save the new session to a local file for future use.
    """
    if Use_Local_Session:
        local_sess = _load_session_from_file(account_no)
        if local_sess is not None:

            # Check if the loaded session is still valid by making a test request to the authentication endpoint
            # logger.info(f"Log In ({account_no})", "Testing local session validity by making a test request...")
            # auth_check = local_sess.get("https://api.worldquantbrain.com/authentication")
            # if auth_check.status_code == 200:
            #     logger.info(f"Log In ({account_no})", "Using local saved session.")
            #     if Return_Permission:
            #         permission_list = auth_check.json().get("permissions", [])
            #         return local_sess, str(tuple(permission_list))
            #     return local_sess
            # logger.info(f"Log In ({account_no})", "Local session is invalid. Re-authenticating...")

            try:
                # Add a timeout so it doesn't hang indefinitely if the network is unstable
                auth_check = local_sess.get("https://api.worldquantbrain.com/authentication", timeout=10)
                
                if auth_check.status_code == 200:
                    logger.info(f"Log In ({account_no})", "Using local saved session.")
                    return local_sess
                
                logger.info(f"Log In ({account_no})", "Local session is invalid. Re-authenticating...")
            
            except requests.exceptions.RequestException as e:
                # Catch SSL, connection, and timeout errors gracefully
                logger.warning(
                    f"Log In ({account_no})", 
                    f"Local session validation failed (Network/SSL error: {e}). Falling back to fresh login..."
                )

    # Proceed with normal login if local session is not used or invalid
    credentials_file_path = os.path.join(CREDENTIALS_PATH, CREDENTIALS_FILE_NAME)
    with open(os.path.expanduser(credentials_file_path), encoding="utf-8") as f:
        credentials = json.load(f)

    if not isinstance(credentials, (list, tuple)) or len(credentials) < 2:
        raise ValueError("Credentials file must contain [username, password]")

    username, password = credentials[0], credentials[1]
    sess = requests.Session()
    sess.auth = HTTPBasicAuth(username, password)

    response = sess.post("https://api.worldquantbrain.com/authentication")
    status_code = response.status_code

    if str(status_code) == ERROR_401:
        logger.error(f"Log In ({account_no})", "❌❌❌Invalid credentials. Please check your credentials file❌❌❌.")
        return None

    logger.info(f"Log In ({account_no})", str(status_code))
    logger.info(f"Log In ({account_no})", str(response.json()))
    # save session as a local file
    _save_session_to_file(sess, account_no)
    return sess

def Check_Session_Timeout(
    SESS,
    threshold: int = 60,
    account_no: str = "N"
):
    """
    - Check if the session is about to expire within the threshold (in minutes).
    - If the session is expired or about to expire, it will re-authenticate and return a new session. 
    Otherwise, it will return the current session.
    - If the input session is None, it will return None.
    """
    if SESS is None:
        logger.info(f"Check Session Timeout ({account_no})", "Session is None...")
        return None

    authentication_url = "https://api.worldquantbrain.com/authentication"
    result = SESS.get(authentication_url)

    if result.status_code == 200:
        result_dict = result.json()
        time_out = result_dict.get("token", {}).get("expiry", {})

        if time_out:
            time_out = time_out / 60
            logger.info(
                f"Check Session Timeout ({account_no})",
                f"Time left for session expiry: {round(time_out, 3)} minutes"
            )
            if time_out < threshold:
                logger.info(f"Check Session Timeout ({account_no})", "re-logging...")
                return Login_to_WQB(account_no=account_no, Use_Local_Session=False)
            return SESS

        logger.info(f"Check Session Timeout ({account_no})", "Session expired, re-authenticating...")
        return Login_to_WQB(account_no=account_no, Use_Local_Session=False)

    logger.error(
        f"Check Session Timeout ({account_no})",
        f"Failed to check session timeout, status code: {result.status_code}. Re-authenticating..."
    )
    return Login_to_WQB(account_no=account_no, Use_Local_Session=False)

def Alpha_IS_Check_New(IS_Check: list):
    """
    - Turn official IS_Check to IS_Check_New for better visualization 
    - (Common Users/Consultants)
    - ❗ if an alpha is already submitted, the is check dict will be like this:
    {"is": {
        "checks": [{
            "name": "ALREADY_SUBMITTED",
            "result": "FAIL"
    }]}}
    - As normal dict, For example:
    "IS-Check": {
        "Status": "FAIL",
        "PASS": {
            "Sharpe": 1.71,
            "Turnover": 0.2402,
            "Matches Competition": [
                "Challenge",
                "International Quant Championship 2025 Stage 1"
            ],
            "PASS-Count": 4
        },
        "FAIL": {
            "Fitness": 0.96,
            "Concentrated Weight": 0.1502,
            "Sub-universe Sharpe": 0.27,
            "FAIL-Count": 3
        },
        "PENDING": {
            "Self-correlation": "None"
        }
    }
    - Special Result:
    {
        'endDate': '2021-01-21',
        'limit': 1.58,
        'name': 'IS_LADDER_SHARPE',
        'result': 'FAIL',
        'startDate': '2023-01-20',
        'value': 0.57,
        'year': 2
    }
    {
        'effective': 2,
        'multiplier': 1.1,
        'name': 'MATCHES_PYRAMID',
        'pyramids': [
            {'multiplier': 1.1, 'name': 'USA/D1/PV'},
            {'multiplier': 1.2, 'name': 'USA/D1/FUNDAMENTAL'}],
        'result': 'PASS'},
    {
        'name': 'MATCHES_THEMES',
        'result': 'WARNING',
        'themes': [
            {'id': 'xD9qQrD', 'multiplier': 2.0, 'name': 'GLB Dataset Utilization Theme'}
        ]
    }
    
    - In super, even baisc pass alpha will own fail contents:
    {'SUPER_SUBMISSION': 1, 'COMBO_DESCRIPTION_LENGTH': 0, 'SELECTION_DESCRIPTION_LENGTH': 0}
    """
    SUBMITTED_DICT = {"name": "ALREADY_SUBMITTED", "result": "FAIL"}
    if SUBMITTED_DICT in IS_Check: return ALREADY_SUBMITTED_ERROR, "❌ This alpha has already been submitted. Please check the IS Check results for details."
    FAIL_EXCEPTION = {'SUPER_SUBMISSION', 'COMBO_DESCRIPTION_LENGTH', 'SELECTION_DESCRIPTION_LENGTH'}
    def Operate_Check_Info(check, DICT):
        NAME = IS_Check_Name_Dict.get(check['name'], 'Unknown')
        if NAME in {"Unknown"}: NAME = check['name'] # set original data
        if NAME in {"Matches Competition"}:
            if check.get("competitions", None): 
                DICT[NAME] = [item['name'] for item in check['competitions']]
        elif NAME in {"IS Ladder Sharpe"}:
            DICT[NAME] = (
                check.get('startDate', 'None'),
                check.get('endDate', 'None'),
                check.get('value', check.get('message', 'None')),
                check.get('limit', 'None'),
            )
        elif NAME in {"Matches Pyramid"}:
            DICT[NAME] = (
                check.get('effective', 'None'),
                check.get('multiplier', 'None'),
                check.get('pyramids', [])
            )
        elif NAME in {"Matches Themes"}: DICT[NAME] = check.get('themes', [])
        else: DICT[NAME] = check.get('value', check.get('message', 'None'))
        return DICT
    PENDING_Count = 0
    FAIL_Count = 0
    PASS_Count = 0
    WARNING_Count = 0
    PENDING = {}
    FAIL = {}
    PASS = {}
    WARNING = {}
    IS_Check_New = {}
    for check in IS_Check:
        result = check["result"]
        name = check['name']
        if name in FAIL_EXCEPTION: result = "WARNING" # change fail exception to warning
        if result in {"PASS"}:
            PASS_Count += 1
            PASS = Operate_Check_Info(check, PASS)
        elif result in {"FAIL"}:
            FAIL_Count += 1
            FAIL = Operate_Check_Info(check, FAIL)
        elif result in {"WARNING"}:
            WARNING_Count += 1
            WARNING = Operate_Check_Info(check, WARNING)
        elif result in {"PENDING"}:
            Name = check['name'] if not IS_Check_Name_Dict.get(check['name'], '') else IS_Check_Name_Dict.get(check['name'])
            PENDING[Name] = "None"
            PENDING_Count += 1
    if FAIL_Count > 0: IS_Check_New["Status"] = "FAIL"
    # elif PENDING_Count > 0: IS_Check_New["Status"] = "Unknown" # PENDING elements will be considered later
    else: IS_Check_New["Status"] = "PASS" # As long as there is not fail element, the alpha can be passed (submission test later)
    PASS["PASS-Count"] = PASS_Count
    FAIL["FAIL-Count"] = FAIL_Count
    WARNING["WARNING-Count"] = WARNING_Count
    PENDING["PENDING-Count"] = PENDING_Count
    IS_Check_New["PASS"] = PASS
    IS_Check_New["FAIL"] = FAIL
    IS_Check_New["PENDING"] = PENDING
    IS_Check_New["WARNING"] = WARNING
    return False, IS_Check_New

def _regular_error_detector(simulation_status: Dict[str, Any]) -> Tuple[bool, str, str]:
    status = simulation_status.get("status", "error")
    if status == "ERROR":
        message = simulation_status.get("message", None)
        if not message:
            return True, str(simulation_status), UNKNOWN_ERROR
        message_lower = message.lower()
        if "invalid data field" in message_lower:
            return True, DATAFIELDS_ERROR, message
        elif "operator" in message_lower:
            return True, OPERATOR_ERROR, message
        elif "variable" in message_lower:
            return True, VARIABLE_ERROR, message
        elif "unexpected character" in message_lower:
            return True, UNEXPECTED_CHARACTER, message
        else:
            return True, UNKNOWN_ERROR, message
    if status == "error":
        return True, UNKNOWN_ERROR, "None"
    if status == "COMPLETE":
        return False, "COMPLETE", "None"
    if status == "WARNING":
        return False, WARNING_ERROR, simulation_status.get("message", "")
    return False, status, "None"

def _get_retry_after(headers: Dict[str, Any]) -> float:
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after is None: return 0.0
    try: return float(retry_after)
    except (TypeError, ValueError): return 0.0

def Get_Self_Corr(SESS, alpha_id: str):
    time.sleep(5)
    while True:
        result = SESS.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}/correlations/self")
        retry_after = _get_retry_after(result.headers)
        if retry_after:
            time.sleep(retry_after)
            continue
        break
    if result.status_code != 200:
        logger.info(f"Get Self Corr ({alpha_id})", f"❌ Error: HTTP {result.status_code}")
        return UNKNOWN_ERROR, f"HTTP {result.status_code}"
    corr_result = result.json()
    if corr_result.get("records", 0) == 0:
        logger.info(f"Get Self Corr ({alpha_id})", "❌ There is no record.")
        return CORR_NO_RECORD_ERROR, "There is no record."
    logger.info(f"Get Self Corr ({alpha_id})", f"✅ Correlation data retrieved successfully with {len(corr_result.get('records', []))} records.")
    return False, corr_result

def Get_Prod_Corr(SESS, alpha_id: str, account_no: str = "N"):
    time.sleep(5)
    while True:
        result = SESS.get(
            f"https://api.worldquantbrain.com/alphas/{alpha_id}/correlations/prod"
        )
        retry_after = _get_retry_after(result.headers)
        if retry_after:
            time.sleep(retry_after)
            continue
        break
    if result.status_code == 200:
        result_dict = result.json()
        if result_dict.get("records", 0) == 0:
            logger.info(f"Get-Prod-Corr ({account_no})", "❌ There is no record.")
            return CORR_NO_RECORD_ERROR, "❌ There is no record."
        logger.info(f"Get-Prod-Corr ({account_no})", f"✅ Correlation data retrieved successfully with {len(result_dict.get('records', []))} records.")
        return False, result_dict
    status_code = result.status_code
    if status_code == 401: return ERROR_401, "❌ Unauthorized access. Please check your session or credentials."
    if status_code == 429: return ERROR_429, "❌ Too many requests. Please try again later."
    logger.info(f"Get-Prod-Corr ({account_no})", f"❌ Error: {status_code}")
    return UNKNOWN_ERROR, "❌ An unknown error occurred."

def Single_Alpha_IS_Summary(
    SESS,
    alpha_id: str,
    account_no: str = "N",
    return_original_data: bool = False,
):
    time.sleep(5)
    api_alpha_yearly_url = (
        f"https://api.worldquantbrain.com/alphas/{alpha_id}/recordsets/yearly-stats"
    )
    alpha_resp = SESS.get(api_alpha_yearly_url)
    if alpha_resp.status_code == 200:
        alpha_resp_dict = dict(alpha_resp.headers)
        content_length = int(alpha_resp_dict.get("Content-Length", 0))
        if not content_length:
            logger.info(f"EVALUATION-IS-Summary ({account_no})", "❌ IS Summary data is empty.")
            return EMPTY_CONTENT_ERROR, "❌ IS Summary data is empty."
        alpha_yearly_data = alpha_resp.json()
        if return_original_data:
            return False, alpha_yearly_data
        yearly_stats = []
        for year_data in alpha_yearly_data.get("records", []):
            year_dict = {
                "Year": year_data[0],
                "Sharpe": year_data[6],
                "Turnover": year_data[5],
                "Fitness": year_data[10],
                "Returns": year_data[7],
                "Drawdown": year_data[8],
                "Margin": year_data[9],
                "Long Count": year_data[3],
                "Short Count": year_data[4],
                "BookSize": year_data[2],
                "pnl": year_data[1],
                "Stage": year_data[11],
            }
            yearly_stats.append(year_dict)
        logger.info(f"EVALUATION-IS-Summary ({account_no})", f"✅ IS Summary data retrieved successfully with {len(yearly_stats)} records.")
        return False, yearly_stats
    status_code = alpha_resp.status_code
    if status_code == 401:
        return ERROR_401, "❌ Unauthorized access. Please check your session or credentials."
    if status_code == 429:
        return ERROR_429, "❌ Too many requests. Please try again later."
    logger.info(f"EVALUATION-IS-Summary ({account_no})", f"❌ Error: {status_code}")
    return UNKNOWN_ERROR, "❌ An unknown error occurred."

def Single_Alpha_Status(
    SESS,
    alpha_id: str,
    alpha_info: Dict[str, Any] = None,
    TYPE: str = "REGULAR",
    return_original_data: bool = False,
    account_no: str = "N"
):
    time.sleep(5)
    api_alpha_url = f"https://api.worldquantbrain.com/alphas/{alpha_id}"
    alpha_resp = SESS.get(api_alpha_url)
    if alpha_resp.status_code == 200:
        alpha_resp_dict = dict(alpha_resp.headers)
        content_length = int(alpha_resp_dict.get("Content-Length", 0))
        if not content_length:
            logger.info(f"EVALUATION-Alpha-Status ({account_no})", f"❌ Empty content in alpha status response. ({alpha_resp_dict})")
            return EMPTY_CONTENT_ERROR, "❌ Status data is empty."
        alpha_data = alpha_resp.json()
        if return_original_data: return False, alpha_data
        alpha_info = alpha_info or {}
        alpha_info["type"] = alpha_data.get("type")
        alpha_info["author"] = alpha_data.get("author")
        settings = alpha_data.get("settings", {}).copy()
        settings["type"] = alpha_data.get("type")
        alpha_info["settings"] = settings
        if TYPE == "REGULAR":
            alpha_info["formula"] = alpha_data.get("regular", {}).get("code", "")
        elif TYPE == "SUPER":
            alpha_info["combo"] = alpha_data.get("combo", {}).get("code", "")
            alpha_info["selection"] = alpha_data.get("selection", {}).get("code", "")
        alpha_info["alpha_id"] = alpha_data.get("id")
        alpha_info["dateCreated"] = alpha_data.get("dateCreated")
        alpha_info["dateModified"] = alpha_data.get("dateModified")
        grade = alpha_data.get("grade")
        is_checks = alpha_data.get("is", {}).get("checks", [])
        fail, is_checks_new = Alpha_IS_Check_New(is_checks)
        if fail: return UNKNOWN_ERROR, is_checks_new
        alpha_info["is_checks"] = is_checks_new
        alpha_info["grade"] = grade
        alpha_info["status"] = alpha_data.get("status")
        logger.info(f"EVALUATION-Alpha-Status ({account_no})", f"✅ Alpha status retrieved successfully. Status: {alpha_info['status']}.")
        return False, alpha_info
    status_code = alpha_resp.status_code
    if status_code == 401:
        return ERROR_401, "❌ Unauthorized access. Please check your session or credentials."
    if status_code == 429:
        return ERROR_429, "❌ Too many requests. Please try again later."
    logger.info(f"EVALUATION-Alpha-Status ({account_no})", f"❌ Error: {status_code}")
    return UNKNOWN_ERROR, "❌ An unknown error occurred."

def Delete_Simulation_Session(SESS, session_id,  account_no="N"):
    time.sleep(1)
    api_url = f"https://api.worldquantbrain.com/simulations/{session_id}"
    alpha_resp = SESS.delete(api_url)
    if alpha_resp.status_code == 200: 
        logger.info(f"Delete-Simulation-Session ({account_no})", f"Session {session_id} Delete Successfully.")
    else:
        Status_Code = alpha_resp.status_code
        logger.error(f"Delete-Simulation-Session ({account_no})", f"Error: {Status_Code}")

def Single_Alpha_Simulation(
    SESS,
    INDEX: int = None,
    TYPE: str = "REGULAR",
    instrumentType: str = "EQUITY",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: int = 0,
    neutralization: str = "INDUSTRY",
    truncation: float = 0.08,
    pasteurization: str = "ON",
    unitHandling: str = "VERIFY",
    nanHandling: str = "OFF",
    language: str = "FASTEXPR",
    visualization: bool = False,
    regular: str = "liabilities / assets",
    selection_regular: str = "color == 'PURPLE'",
    combo_regular: str = "1",
    selectionHandling: str = "POSITIVE",
    selectionLimit: int = 10,
    componentActivation: str = "IS",
    testPeriod: str = "P1Y",
    maxTrade: str = "ON",
    Settings_Dict: Dict[str, Any] = None,
    Folder_Path: str = "",
    File_Name: str = None,
    account_no: str = "N"
):
    time.sleep(5)
    if not Settings_Dict:
        simulation_data = {
            "type": TYPE,
            "settings": {
                "instrumentType": instrumentType,
                "region": region,
                "universe": universe,
                "delay": delay,
                "decay": decay,
                "neutralization": neutralization,
                "truncation": truncation,
                "pasteurization": pasteurization,
                "unitHandling": unitHandling,
                "nanHandling": nanHandling,
                "language": language,
                "visualization": visualization,
                "testPeriod": testPeriod,
                "maxTrade": maxTrade,
            },
        }
        if TYPE == "SUPER":
            simulation_data["settings"].update(
                {
                    "selectionHandling": selectionHandling,
                    "selectionLimit": selectionLimit,
                    "componentActivation": componentActivation,
                }
            )
            simulation_data.update(
                {"combo": combo_regular, "selection": selection_regular}
            )
        else: simulation_data["regular"] = regular
    else: simulation_data = Settings_Dict
    logger.info(f"SIMULATION ({account_no})", f"Begin simulation...")
    sim_resp = SESS.post(
        "https://api.worldquantbrain.com/simulations", json=simulation_data
    )
    sim_resp_dict = dict(sim_resp.headers)
    sim_progress_url = sim_resp_dict.get("Location", None)
    if not sim_progress_url:
        sim_resp_json = sim_resp.json()
        return UNKNOWN_ERROR, str(sim_resp_json)

    # Display rate limit info
    x_rate_limit = sim_resp_dict.get("X-Ratelimit-Limit", "None")
    x_rate_remaining = sim_resp_dict.get("X-Ratelimit-Remaining", "None")
    x_rate_reset = sim_resp_dict.get("X-Ratelimit-Reset", "None")
    reset_seconds = None
    try: reset_seconds = float(x_rate_reset)
    except (TypeError, ValueError): reset_seconds = None
    if reset_seconds is None: reset_text = f"{x_rate_reset}s"
    else:
        reset_text = f"{reset_seconds:.2f}s ({reset_seconds / 3600:.2f} h)"
    logger.info(
        f"SIMULATION ({account_no})",
        (
            f"🚨Rate Limit: {x_rate_limit}, Remaining: {x_rate_remaining}, "
            f"Reset Time: {reset_text}"
        ),
    )

    session_id = sim_progress_url.split("/")[-1]
    logger.info(f"SIMULATION ({account_no})", f"Session ID: {session_id}")
    time.sleep(1)
    start_time = time.time()
    timeout_sec = 1200
    request_timeout = 420
    request_timeout_num = 0
    sim_progress_resp = UNKNOWN_ERROR

    new_progress = progress = 0
    while True:
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time >= timeout_sec:
            sim_progress_resp = SIMULATION_RETRY_TIMEOUT_ERROR
            break
        try:
            sim_progress_resp = SESS.get(sim_progress_url, timeout=request_timeout)
            sim_progress_resp_dict = dict(sim_progress_resp.headers)
            if float(sim_progress_resp_dict.get("Content-Length", 0)) > 0: new_progress = sim_progress_resp.json().get("progress", 0)
            else: logger.info(f"SIMULATION ({account_no})", f"🔥 Progress: {sim_progress_resp_dict}")
            if new_progress != progress: 
                logger.info(f"SIMULATION ({account_no})", f"🔥 Progress: {new_progress}")
                progress = new_progress
        except requests.exceptions.Timeout:
            request_timeout_num += 1
            continue
        except requests.exceptions.RequestException as e:
            sim_progress_resp = e
            break
        retry_after_sec = _get_retry_after(sim_progress_resp.headers)
        if retry_after_sec == 0: break
        
        elapsed_time = time.time() - start_time
        if elapsed_time + retry_after_sec >= timeout_sec:
            sim_progress_resp = SIMULATION_RETRY_TIMEOUT_ERROR
            break
        else: time.sleep(retry_after_sec)
    if session_id: Delete_Simulation_Session(SESS, session_id, account_no=account_no)

    if request_timeout_num:
        logger.info(
            f"SIMULATION ({account_no})",
            f"Polling encountered {request_timeout_num} request timeouts."
        )

    if sim_progress_resp in {
        SIMULATION_RETRY_TIMEOUT_ERROR,
        SIMULATION_REQUEST_TIMEOUT_ERROR,
        UNKNOWN_ERROR,
    }:
        logger.info(
            f"SIMULATION ({account_no})",
            (
                "Timeout Error due to "
                f"{sim_progress_resp} (Request timeout num: {request_timeout_num})"
            )
        )
        return UNKNOWN_ERROR, sim_progress_resp

    alpha_info_dict = sim_progress_resp.json()
    error, info, message = _regular_error_detector(alpha_info_dict)
    if error:
        logger.info(f"SIMULATION ({account_no})", f"Error due to {info} (Message: {message})")
        return info, message
    if info == WARNING_ERROR: logger.info(f"SIMULATION ({account_no})({INDEX})", f" ⛔Warning: {message}")
    alpha_id = alpha_info_dict.get("alpha")
    if alpha_id: alpha_info_dict.setdefault("settings", {}).update({"type": TYPE})
    logger.info(f"SIMULATION ({account_no})({INDEX})", f"({File_Name}) {alpha_id} generated.")
    return False, alpha_info_dict

def simulate_and_evaluate_alpha(
    alpha_settings,
    alpha_id="",
    regular="",
    selection="",
    combo="",
    sess=None,
    account_no="N",
    include_self_corr=True,
    include_prod_corr=True,
):
    """
    Run simulation, evaluation, and correlation for a single alpha.

    Parameters
    ----------
    alpha_settings : dict
        Input settings dict (see README), e.g.
        {"type": "REGULAR", "settings": {...}}.
    regular : str
        Regular formula for REGULAR alphas. Can also be provided in alpha_settings["regular"].
    selection : str
        Selection formula for SUPER alphas. Optional if provided in alpha_settings["selection"].
    combo : str
        Combo formula for SUPER alphas. Optional if provided in alpha_settings["combo"].
    sess : requests.Session
        Optional session to reuse. If not provided, credentials_path is required.

    Returns
    -------
    (bool, dict | str)
        Success flag and result dict, or error message on failure.
    """
    if sess is None:
        sess = Login_to_WQB(account_no=account_no)
        if sess is None:
            return UNKNOWN_ERROR, "Login failed."
    else:
        sess = Check_Session_Timeout(sess, account_no=account_no)
        if sess is None:
            return UNKNOWN_ERROR, "Session expired and re-login failed."
    if not regular and alpha_settings.get("type", "REGULAR") == "REGULAR":
        regular = alpha_settings.get("regular", "")
    logger.info(f"SIMULATION ({account_no})", f"Starting simulation and evaluation process...")
    logger.info(f"SIMULATION ({account_no})", f"Regular formula: {regular}")
    logger.info(f"SIMULATION ({account_no})", f"Simulation settings: {alpha_settings}")
    sim_info = {}
    if not alpha_id:
        fail, sim_info = _retry_after(
            FUNCTION=Single_Alpha_Simulation,
            SESS=sess,
            TYPE=alpha_settings.get("type", "REGULAR"),
            Settings_Dict=alpha_settings,
            account_no=account_no
        )
        if fail: return False, sim_info

        alpha_id = sim_info.get("alpha") or sim_info.get("alpha_id")
        if not alpha_id: return UNKNOWN_ERROR, "Simulation succeeded but alpha id was missing."
    
    # Simulation info output, including status and message for regular errors, and other info for successful simulation.
    sim_info_output = {}
    simulation_status = sim_info.get("status", "")
    simulation_message = sim_info.get("message", "")
    sim_info_output["status"] = simulation_status
    sim_info_output["message"] = simulation_message
    logger.info(f"SIMULATION-INFO ({account_no})", f"Simulation status: {simulation_status}, output: {sim_info_output}")

    fail, status_info = _retry_after(
        FUNCTION=Single_Alpha_Status,
        SESS=sess,
        alpha_id=alpha_id,
        TYPE=alpha_settings.get("type", "REGULAR"),
        account_no=account_no
    )
    if fail: return UNKNOWN_ERROR, status_info

    evaluation_status = status_info.get("status", "")
    evaluation_is_checks = status_info.get("is_checks", [])
    # evaluation_settings = status_info.get("settings", {})

    # fail, is_summary = _retry_after(
    #     FUNCTION=Single_Alpha_IS_Summary,
    #     SESS=sess,
    #     alpha_id=alpha_id,
    #     account_no=account_no
    # )
    # if fail: return UNKNOWN_ERROR, is_summary

    evaluation = {
        # "settings": evaluation_settings,
        "status": evaluation_status,
        "is_checks": evaluation_is_checks,
        # "is_summary": is_summary,
    }
    logger.info(f"EVALUATION ({account_no})", f"Evaluation status: {evaluation_status}, output: {evaluation}")

    correlation = {}
    if include_self_corr:
        self_fail, self_corr = _retry_after(
            FUNCTION=Get_Self_Corr,
            SESS=sess,
            alpha_id=alpha_id,
        )
        correlation["self_corr"] = self_corr if not self_fail else {"error": self_corr}
    if include_prod_corr:
        prod_fail, prod_corr = _retry_after(
            FUNCTION=Get_Prod_Corr,
            SESS=sess,
            alpha_id=alpha_id,
            account_no=account_no
        )
        correlation["prod_corr"] = prod_corr if not prod_fail else {"error": prod_corr}

    logger.info(f"CORRELATION ({account_no})", f"Correlation results: {correlation}")
    return False, {
        "simulation": sim_info_output,
        "evaluation": evaluation,
        "correlation": correlation,
    }

def main(regular, settings, account_no="0"):
    # account_no = "0"
    # regular = "rank(ts_mean(close, 10) - ts_mean(close, 50))"
    # settings = {
    #     "type": "REGULAR",
    #     "instrumentType": "EQUITY",
    #     "region": "MEA",
    #     "universe": "TOP400",
    #     "delay": 1,
    #     "decay": 3,
    #     "neutralization": "SUBINDUSTRY",
    #     "truncation": 0.08,
    #     "pasteurization": "ON",
    #     "nanHandling": "OFF",
    #     "testPeriod": "P1Y",
    #     "maxTrade": "ON",
    #     "maxPosition": "OFF",
    #     "regular": regular,
    # }
    
    # Step 1: Initialize global variables and build simulation payload
    fail, info = initialize_global_variables(account_no=account_no)
    if fail: return info
    
    # Step 2: Build simulation payload
    fail, simulation_payload = _build_simulation_payload(settings, regular, account_no=account_no)
    if fail: return simulation_payload
    
    # Step 3: Validate regular formula and its compatibility with settings
    fail, validation_result = _validate_regular_formula(
        regular_formula=regular, 
        region=simulation_payload.get("settings", {}).get("region"),
        delay=simulation_payload.get("settings", {}).get("delay"), 
        universe=simulation_payload.get("settings", {}).get("universe"), 
        account_no=account_no
    )
    if fail: return validation_result
    
    # Step 4: Log in to WQB and run simulation, evaluation, and correlation
    SESS = Login_to_WQB(account_no=account_no)
    fail, result = simulate_and_evaluate_alpha(
        alpha_settings=simulation_payload,
        regular=regular, # Optional if already included in simulation_payload
        account_no=account_no,
        sess=SESS,  # Optional: provide existing session to reuse
    )
    if not fail: logger.info("Main", result)
    else: return result

# Determine if we are running standalone or being imported
if __name__ == "__main__":
    # Run independently: create its own separate log file
    logger = setup_logger(LOG_DIR, "alpha_simulator", logger_obj_name="alpha_simulator_logger")
    account_no = "0"
    regular = "rank(ts_mean(close, 10) - ts_mean(close, 50))"
    settings = {
        "type": "REGULAR",
        "instrumentType": "EQUITY",
        "region": "MEA",
        "universe": "TOP400",
        "delay": 1,
        "decay": 3,
        "neutralization": "SUBINDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "nanHandling": "OFF",
        "testPeriod": "P1Y",
        "maxTrade": "ON",
        "maxPosition": "OFF",
        "regular": regular,
    }
    main(regular=regular, settings=settings, account_no=account_no)
else:
    # Imported by Script A: Fetch A's underlying logger by name and wrap it
    # so the .info(flag, msg) syntax still works flawlessly!
    underlying_logger = logging.getLogger("shared_logger")
    logger = FlagLogger(underlying_logger)