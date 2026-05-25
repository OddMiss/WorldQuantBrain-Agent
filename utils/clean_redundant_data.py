import re

def clean_community_data(text):
    # Phrases commonly found in UI text that we want to discard
    boilerplate_blacklist = [
        "comments",
        "be the first to write a comment",
        "search",
        "platform community",
        "faq submit a request",
        "english (united states)",
        "sort by",
        "worldquant brain",
        "getting started with research",
        "Hello, Community!",
        "BRAIN®"
    ]
    
    cleaned_lines = []
    
    # Split text line by line
    for line in text.split('\n'):
        line_stripped = line.strip()
        
        # 1. Skip completely empty lines
        if not line_stripped:
            continue
            
        # 2. Skip lines that are just numbers (like standalone 0 or 18)
        if line_stripped.isdigit():
            continue
            
        # 3. Skip UI text matching "Follow 1", "Follow 11", etc.
        if re.match(r'^follow\s+\d+', line_stripped, re.IGNORECASE):
            continue
            
        # 4. Skip timestamps like "3 months ago", "4 days ago"
        if re.search(r'\d+\s+(month|day|week|year)s?\s+ago', line_stripped, re.IGNORECASE):
            continue
        
        # 5. Remove lines that are just a string like "JG63643" (two capitals first and then five numbers)
        if re.match(r'^[A-Z]{2}\d{5}$', line_stripped):
            continue
        
        # 6. Skip lines that contain any of our blacklisted boilerplate phrases
        should_skip = False
        for phrase in boilerplate_blacklist:
            if phrase in line_stripped.lower():
                should_skip = True
                break
                
        if should_skip:
            continue
            
        # If it passes all filters, save the cleaned line
        cleaned_lines.append(line_stripped)

    # Join the lines back together
    full_output = "\n".join(cleaned_lines)

    # Join everything back together
    return re.sub(r'\n\s*\n+', '\n', full_output).strip()

if __name__ == "__main__":
    # --- Your Raw Data ---
    raw_input_data = """
    0 

    Comments

    0 comments

    Be the first to write a comment.

    Search

    Follow 1

    BRAINplatform Community FAQ Submit a request HH11690 English (United States)



    BX86068 

    3 months ago

    thanks,it worth a try 

    0 

    BRAINplatform Community FAQ Submit a request HH11690 English (United States)



    18

    Comments

    2 comments

    Search

    Follow 11 

    Sort by 

    SW93074 

    4 months ago



    WorldQ uant BRAIN > Community> Getting started with Research
    """

    # Run the script
    output = clean_community_data(raw_input_data)
    print(output)