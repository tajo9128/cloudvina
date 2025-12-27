import logging
import re
from supabase import Client

logger = logging.getLogger("db_utils")

def safe_update(db: Client, table: str, match_criteria: dict, update_data: dict, schema_cache_retry=True):
    """
    Executes a Supabase update with automatic handling of 'PGRST204' (Column Missing) errors.
    
    Args:
        db: Supabase client
        table: Table name (e.g., 'jobs')
        match_criteria: Dictionary of exact matches (e.g., {'id': '123'})
        update_data: Dictionary of fields to update
        schema_cache_retry: Recursive safety flag
        
    Returns:
        The result of the execute() call.
    """
    if not update_data:
        return None

    try:
        query = db.table(table).update(update_data)
        for key, value in match_criteria.items():
            query = query.eq(key, value)
            
        return query.execute()

    except Exception as e:
        # Check for Postgrest API Error
        # postgrest-py raises APIError which usually has a .code or dictionary message
        error_dict = getattr(e, 'args', [{}])[0] if isinstance(e.args, tuple) and len(e.args) > 0 else {}
        if isinstance(error_dict, str): # Sometimes it's a string
            try: error_dict = eval(error_dict)
            except: pass
            
        # Get code/message safely
        code = getattr(e, 'code', None) or error_dict.get('code')
        message = getattr(e, 'message', None) or error_dict.get('message', str(e))

        # Handle Schema Cache / Missing Column Error
        if code == 'PGRST204' or "Could not find the" in str(message):
            # Extract column name using regex
            # Message format: "Could not find the 'notes' column of 'jobs' in the schema cache"
            match = re.search(r"Could not find the '(\w+)' column", str(message))
            if match:
                missing_col = match.group(1)
                logger.warning(f"⚠️ safe_update: Dropping missing column '{missing_col}' from update to table '{table}'")
                
                # Create clean copy
                clean_data = update_data.copy()
                if missing_col in clean_data:
                    del clean_data[missing_col]
                
                # Retry recursively if we still have data to update
                if clean_data:
                    return safe_update(db, table, match_criteria, clean_data, schema_cache_retry)
                else:
                    logger.warning(f"⚠️ safe_update: No fields left to update for '{table}' after cleaning.")
                    return None
        
        # If not handled, re-raise
        raise e
