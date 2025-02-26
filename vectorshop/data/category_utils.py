def get_primary_category(category_str: str) -> str:
    """
    Given a full category string (e.g., "Computers&Accessories|Accessories&Peripherals|..."),
    return the primary category (i.e. the first part).
    """
    if not isinstance(category_str, str):
        return ""
    return category_str.split("|")[0].strip().lower()

def get_category_info(category_str: str, levels: int = 2) -> str:
    """
    Extract a summary of the category information from a full category string.
    
    Args:
        category_str (str): Full category string delimited by "|"
        levels (int): Number of levels to include (default is 2).
        
    Returns:
        str: A string that concatenates the primary category and subcategory levels.
    """
    if category_str:
        parts = [part.strip() for part in category_str.split("|")]
        # Include up to the specified number of levels, but at least the primary category.
        return " | ".join(parts[:max(1, levels)])
    return ""

