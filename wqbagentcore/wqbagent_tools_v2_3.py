from crewai.tools import tool

# ====================== DOCS SEARCH TOOL ======================
@tool("retrieve_text_data")
def retrieve_text_data(query: str) -> str:
    """Fetches relevant text snippets based on a string query.
    Input a highly specific financial concept or math operator (e.g., 'supply chain momentum', 'analyst revision').
    Returns text context to be used for answering user queries."""
    
    docs = combined_retriever.invoke(query)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# ====================== JSON SEARCH TOOLS ======================
@tool("search_operators")
def search_operators(query: str) -> str:
    """Search the operator definitions and syntax. 
    Input a concept or specific operator name."""
    results = []
    query_lower = query.lower()
    for op_name, op_details in operators_data.items():
        if (query_lower in op_name.lower() or 
            query_lower in op_details.get('description', '').lower() or 
            query_lower in op_details.get('category', '').lower()):
            
            res = f"Operator: {op_name}\nSyntax: {op_details.get('definition')}\nDesc: {op_details.get('description')}"
            results.append(res)
            
        if len(results) >= 15:  # Limit results to save context window
            break
            
    return "\n---\n".join(results) if results else "No matching operators found."

@tool("search_datafields")
def search_datafields(query: str) -> str:
    """Search the data dictionary for dataset fields. 
    Input a concept or specific field name."""
    results = []
    query_lower = query.lower()
    for field_name, field_details in datafields_data.items():
        if (query_lower in field_name.lower() or 
            query_lower in field_details.get('description', '').lower() or 
            query_lower in field_details.get('category_name', '').lower()): # Note: matching your JSON typo 'category_name'
            
            abbr_region_list = field_details.get('region', [])
            full_region_list = [Abbr_To_Full.get(abbr, abbr) for abbr in abbr_region_list]
            res = f"Field: {field_name}\nType: {field_details.get('type')}\nDesc: {field_details.get('description')}\nRegion(s): {', '.join(full_region_list)}"
            results.append(res)
            
        if len(results) >= 15:  # Limit results to save context window
            break
            
    return "\n---\n".join(results) if results else "No matching data fields found."

# 🚨 Instead of forcing the agent to memorize the entire global rulebook upfront in the docstring, 
# you should provide a dedicated, lightweight lookup tool.
# When the coder or validator decides to work on a specific region (like "USA" or "CHN"), 
# it will call this lookup tool to fetch only the rules for that specific region. This keeps 
# your context window perfectly clean until the exact moment the rules are needed.

@tool("get_region_allowed_settings")
def get_region_allowed_settings(region: str) -> str:
    """
    Returns the exact allowed universes, delays, and neutralization rules for a specific region.
    Call this BEFORE generating settings or formulas for a region to ensure parameters are valid.
    
    Input:
    - region: A 3-letter region string, "USA", "GLB", "EUR", "ASI", "CHN", "IND", "KOR", "TWN", "MEA"
    """
    rules = {
        "USA": {
            "delay": [0, 1],
            "universe": ["TOP3000", "TOP2000", "TOP1000", "TOP500", "TOP200", "TOPSP500", "ILLIQUID_MINVOL1M"],
            "neutralization": ["SECTOR", "INDUSTRY", "SUBINDUSTRY"]
        },
        "GLB": {
            "delay": [1],
            "universe": ["TOP3000", "MINVOL1M", "MINVOL10M", "TOPDIV3000"],
            "neutralization": ["SECTOR", "INDUSTRY", "COUNTRY"]
        },
        "EUR": {
            "delay": [0, 1],
            "universe": ["TOP2500", "TOP1200", "TOP800", "TOP400", "ILLIQUID_MINVOL1M", "TOPCS1600"],
            "neutralization": ["SECTOR", "INDUSTRY"]
        },
        "ASI": {
            "delay": [1],
            "universe": ["TOP500", "MINVOL1M", "MINVOL10M", "ILLIQUID_MINVOL1M"],
            "neutralization": ["SECTOR", "INDUSTRY", "COUNTRY"]
        },
        "CHN": {
            "delay": [0, 1],
            "universe": ["TOP2000U"],
            "neutralization": ["SECTOR", "INDUSTRY"]
        },
        "IND": {"delay": [1], "universe": ["TOP500"], "neutralization": ["SECTOR", "INDUSTRY"]},
        "KOR": {"delay": [1], "universe": ["TOP600"], "neutralization": ["SECTOR", "INDUSTRY"]},
        "TWN": {"delay": [1], "universe": ["TOP500", "TOP100"], "neutralization": ["SECTOR", "INDUSTRY"]},
        "MEA": {"delay": [1], "universe": ["TOP400", "TOP300"], "neutralization": ["SECTOR", "INDUSTRY"]}
    }
    
    region_upper = region.upper().strip()
    if region_upper in rules:
        return f"Allowed configuration for region {region_upper}:\n{json.dumps(rules[region_upper], indent=2)}"
    else:
        return f"❌ Region '{region}' not recognized. Choose from: {list(rules.keys())}"

@tool("check_regular_formula")
def check_regular_formula(regular_formula: str, region: str, delay: int, universe: str) -> str:
    """Validates alpha syntax and data fields locally. Ensure you check 'get_region_allowed_settings' first to avoid invalid combinations."""
    try:
        success, result = _validate_regular_formula(
            regular_formula=regular_formula,
            region=region,
            delay=int(delay),
            universe=universe,
            account_no=account_no # global variable set during initialization
        )
        if not success:
            return f"❌ Validation Failed: {result}\nReview the datafields and operators used."
        return f"✅ Formula Check Passed! Multiplier details: {result}"
    except Exception as e:
        return f"❌ Checker Exception: {e}"

@tool("wqb_simulate_api")
def wqb_simulate_api(settings: str, regular_formula: str) -> str:
    """
    Run full alpha simulation and evaluation by calling the WQB API.
    
    Inputs:
    - settings: JSON string of simulator settings. Must included keys: [
        "instrumentType", # default "EQUITY"
        "region", # "USA", "GLB", "EUR", "ASI", "CHN", "IND", "KOR", "TWN", "MEA"
        "universe", # depends on region, e.g. "TOP3000", "MINVOL1M", etc.
        "delay", # 0 or 1, depends on region
        "decay", # between 0 and 512, suggest to be integer for better performance, but float is also acceptable
        "neutralization", # depends on region, e.g. "SECTOR", "INDUSTRY", "COUNTRY", "GICS_SECTOR"
        "truncation", # between 0 and 1, inclusive, float only, suggest keep 2 decimal places for better performance, but more decimal places are also acceptable
        "pasteurization", # either ON or OFF
        "nanHandling", # either ON or OFF
        "testPeriod", # Format 1: with year and without month, like "P1Y"; Format 2: with year and month, like "P1Y2M"; Format 3: with month only, like "P2M"
        "maxTrade", # either ON or OFF
        "maxPosition" # either ON or OFF
        # Max Position and Max Trade cannot both be set to On simultaneously
    ]
    - regular_formula: The WorldQuant expression string.
    
    Output:
    Returns the JSON payload containing Simulation Status, IS_Checks, and Correlation.
    If the status is not 'COMPLETE' or IS_Checks fail, read the feedback and try again!
    """
    try:
        settings_payload = json.loads(settings) if isinstance(settings, str) else settings
    except json.JSONDecodeError as e:
        return f"❌ Invalid settings JSON: {e}"

    # 1. Build Payload and validate settings exactly as the simulator expects
    success, sim_payload = _build_simulation_payload(settings_payload, regular_formula, account_no="0")
    if not success:
        return f"❌ Payload Build Error: {sim_payload}"

    # 2. Run simulation and evaluation
    success, result = simulate_and_evaluate_alpha(
        alpha_settings=sim_payload,
        regular=regular_formula,
        account_no=account_no, # global variable set during initialization
        include_self_corr=True,
        include_prod_corr=True
    )
    
    if not success:
        return f"❌ API Request Failed: {result}"
        
    return result