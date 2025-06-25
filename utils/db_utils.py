import duckdb

def QUERY(query, conn = None, read_only = True):
    # Check if a connection is provided
    if conn:
        # Execute the query
        result = conn.sql(query)
        if result is not None:
            result = result.df()
        # Return the result
        return result
    else:
        # Connect to DuckDB
        with duckdb.connect(
            database = '~/LocalData/database.db',
            read_only = read_only,
        ) as conn:
            # Execute the query
            result = conn.sql(query)
            if result is not None:
                result = result.df()

            conn.close()
                
            # Return the result
            return result

