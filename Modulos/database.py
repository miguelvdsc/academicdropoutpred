import datetime
import json
import os
import joblib
import numpy as np
import pandas as pd
import sqlalchemy as sql
import bcrypt
from sqlalchemy.exc import SQLAlchemyError

def login_query(id):
    """
    Retrieves a user's name and type from a PostgreSQL database.

    Parameters:
    - id (int): The ID of the user to retrieve.

    Returns:
    - name (str): The name of the user.
    - tipo (str): The type of the user.
    """
    
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        query = sql.text(f"SELECT name, tipo FROM users WHERE id = {id}")
        result = conn.execute(query)
        data = result.fetchall()
        name=data[0][0]
        tipo=data[0][1]
        return name,tipo
    
    
def check_if_user_exists(username,password):
    """
    Retrieves a user's ID from a PostgreSQL database.

    Parameters:
    - username (str): The username of the user to retrieve.

    Returns:
    - id (int): The ID of the user if the user exists.
    - False (bool): False if the user does not exist.
    """
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        query = sql.text(f"SELECT id FROM users WHERE name = '{username}'")
        result = conn.execute(query)
        data = result.fetchall()
        if len(data)==0:
            return False
        else:
            return data[0][0]
        
def register_user(name, email, password, cargo):
    """
    Registers a new user in a PostgreSQL database.

    Parameters:
    - name (str): The name of the user.
    - email (str): The email of the user.
    - password (str): The password of the user.
    - cargo (str): The role of the user.

    Returns:
    - True (bool): True if the user was successfully registered.
    - False (bool): False if an error occurred.

    """
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        try:
            query = sql.text("INSERT INTO users (name, email, password, tipo) VALUES (:name, :email, :hashed_password, :cargo)")
            params = {'name': name, 'email': email, 'hashed_password': hashed_password.decode('utf-8'), 'cargo': cargo}
            result = conn.execute(query, params)
            conn.commit()
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
    
def change_password(id, password):
    """
    Updates a user's password in a PostgreSQL database.

    Parameters:
    - id (int): The ID of the user.
    - password (str): The new password of the user.

    Returns:
    - True (bool): True if the password was successfully updated.
    - False (bool): False if an error occurred.
    """
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        try:
            query = sql.text("UPDATE users SET password = :hashed_password WHERE id = :id")
            params = {'hashed_password': hashed_password.decode('utf-8'), 'id': id}
            result = conn.execute(query, params)
            conn.commit()
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

def check_pass(id, password):
    """
    Check if the provided password matches the password stored in the database for the given user ID.

    Args:
        id (int): The user ID.
        password (str): The password to check.

    Returns:
        bool: True if the password matches, False otherwise.
    """
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        query = sql.text(f"SELECT password FROM users WHERE id = {id}")
        result = conn.execute(query)
        data = result.fetchall()
        if bcrypt.checkpw(password.encode('utf-8'), data[0][0].encode('utf-8')):
            return True
        else:
            return False
    
    
def modify_estado(id):
    """
    Modifies the estado (state) of a user in the database.

    Args:
        id (int): The ID of the user to modify.

    Returns:
        None
    """
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        query = sql.text("UPDATE users SET estado = 0 WHERE id = :id")
        params = {'id': id}
        result = conn.execute(query, params)
        conn.commit()
        
        
def select_from_table(table):
    """
    Retrieve all rows from the specified table in the database.

    Args:
        table (str): The name of the table to select from.

    Returns:
        tuple: A tuple containing two elements:
            - columns (list): A list of column names in the selected table.
            - data (list): A list of tuples representing the rows of data from the selected table.
    """
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        query = sql.text(f"SELECT * FROM {table}")
        result = conn.execute(query)
        columns = result.keys()
        data = result.fetchall()
        return columns, data
    
def store_dataset(dataset,name,tipo):
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        try:
            df = pd.read_csv(dataset)
            df.drop(columns=['Unemployment rate', 'Inflation rate', 'GDP'], inplace=True)
            
            query = sql.text("INSERT INTO dataset (name, tipo) VALUES (:name, :tipo) RETURNING id_dataset")
            params = {'name': name, 'tipo': tipo}
            result = conn.execute(query, params)
            id = result.fetchone()[0]
            
            # Add a new column to the DataFrame with the ID
            df['id_dataset'] = id
            
            # Insert the DataFrame into the 'dataset_atributos' table
            df.to_sql('dataset_atributos', conn, if_exists='append', index=False)
            
            conn.commit()
        except Exception as e:
            print(f"An error occurred: {e}")
            
            
def select_head_dataset(id):
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        query = sql.text(f"SELECT * FROM dataset_atributos WHERE id_dataset={id} LIMIT 5")
        result = conn.execute(query)
        columns = result.keys()
        data = result.fetchall()
        return data,columns
    
def select_from_table_id_one_dataset(table,id):
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        query = sql.text(f"SELECT * FROM {table} WHERE id_dataset = {id}")
        result = conn.execute(query)
        columns = result.keys()
        data = result.fetchall()
        return data,columns
    
def select_from_table_dataset_type(type):
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        query = sql.text(f"SELECT * FROM dataset WHERE tipo = '{type}'")
        result = conn.execute(query)
        columns = result.keys()
        data = result.fetchall()
        return columns, data
    
    
def insert_into_model(nome,parametros,id_dataset):
    """
    Inserts a new row into the 'model' table in the PostgreSQL database.

    Parameters:
    name (str): The name to be inserted into the 'name' column.
    data (str): The data to be inserted into the 'data' column.
    parametros (dict): The dictionary to be converted to JSON and inserted into the 'parametros' column.

    Returns:
    None
    """
    # Create a connection to the PostgreSQL database
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    autor="admin"
    # Connect to the database
    with engine.connect() as conn:
        # Define the SQL query
        query = sql.text("INSERT INTO modelo (nome, parametros, autor,filename,id_dataset) VALUES (:nome, :parametros, :autor, :filename, :id_dataset) RETURNING id_modelo")

        # Define the parameters
        params = {'nome': nome, 'parametros': json.dumps(parametros), 'autor': autor, 'filename': None, 'id_dataset': id_dataset}

        # Execute the query and fetch the returned id
        result = conn.execute(query, params)
        inserted_id = result.fetchone()[0]
        
        conn.commit()

    # Now 'inserted_id' contains the ID of the inserted row
    return inserted_id
        
def query_to_dataframe(table, column, value):
    """
    Executes a SQL query on a PostgreSQL database and returns a pandas DataFrame.

    Parameters:
    table (str): The name of the table in the database.
    column (str): The name of the column to filter on.
    value (str, int, float, etc.): The value to filter on.

    Returns:
    pd.DataFrame: A DataFrame that includes the rows where the value in the specified column equals the provided value.
    """

    # Create a connection to the PostgreSQL database
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

    # Define your SQL query
    query = sql.text(f"SELECT * FROM {table} WHERE {column} = :value")

    # Execute the query and convert the result to a DataFrame
    df = pd.read_sql_query(query, engine, params={"value": value})

    return df


def save_model(model, filename):
    """
    Saves a trained model to a specified filepath.

    Parameters:
    model (sklearn.base.BaseEstimator): The trained model.
    directory (str): The directory where the model should be saved.
    filename (str): The name of the file.

    Returns:
    None
    """
    directory="static/downloads"
    filepath = os.path.join(directory, filename)
    joblib.dump(model, filepath)
    
def update_model_file_database(id, filename):
    """
    Updates the 'file' column in the 'model' table in the PostgreSQL database.

    Parameters:
    id (int): The ID of the model to update.
    filename (str): The new filename to be inserted into the 'file' column.

    Returns:
    None
    """
    # Create a connection to the PostgreSQL database
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

    # Connect to the database
    with engine.connect() as conn:
        # Define the SQL query
        query = sql.text("UPDATE modelo SET filename = :filename WHERE id_modelo = :id")

        # Define the parameters
        params = {'filename': filename, 'id': id}

        # Execute the query
        conn.execute(query, params)
        
        conn.commit()
        
def update_model_file(model_id,model):
    """
     Update a model file using info from the database.

    Parameters:
    id (int): The ID of the model to update.
    filename (str): The new filename to be inserted into the 'file' column.

    Returns:
    None
    """
    
    try:
        # first retrieve the model info
        model_info = retrieve_model_info('modelo', model_id)
        model_name = model_info[0][1]
        print(model_name)
        file_name = model_name + '_' + str(model_id) + '.pkl'
        save_model(model, file_name)
        update_model_file_database(model_id, file_name)
        return True
    except SQLAlchemyError as e:
        print(f"An error occurred while updating the model file for id {model_id} .")
        print(str(e))
        return False
    
def retrieve_model_info(table, id_modelo):
    """
    Retrieves information about a specific model from a database table.

    Parameters:
    table (str): The name of the table in the database.
    id_modelo (int): The ID of the model.

    Returns:
    list: A list of tuples where each tuple represents a row in the result set.
    """

    # Create a connection to the PostgreSQL database
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

    # Connect to the database
    with engine.connect() as conn:
        # Define the SQL query
        query = sql.text(f"SELECT * FROM {table} WHERE id_modelo = {id_modelo}")

        # Execute the query
        result = conn.execute(query)

        # Fetch all rows from the result set
        data = result.fetchall()

    # Return the data
    return data
    
# def store_evaluation_ant(id_model,accuracy,precision,recall,roc_auc,f1,tn,fp,fn,tp):
#     """
#     Stores the evaluation results in the 'evaluation' table in the PostgreSQL database.

#     Parameters:
#     model_id (int): The ID of the model.
#     matrix (list): The confusion matrix to be stored.

#     Returns:
#     - success (bool): True if the evaluations were stored successfully, False otherwise.
#     """
#     # flat_matrix = matrix.ravel()
#     # print(flat_matrix)
#     # # Check the size of the flattened matrix
#     # if len(flat_matrix) != 4:
#     #     print(f"Expected a 2x2 confusion matrix, but got a matrix with {len(flat_matrix)} elements.")
#     #     return False

#     # fp, fn, tp, tn = matrix.ravel()
    
#     # # Convert numpy.int64 types to int
#     # fp, fn, tp, tn = int(fp), int(fn), int(tp), int(tn)
    
#     try:
#         # Create a connection to the PostgreSQL database
#         engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

#         # Connect to the database
#         with engine.connect() as conn:
#             # Define the SQL query
#             query = sql.text("INSERT INTO avaliacao (tp,tn,fp,fn,id_modelo) VALUES (:tp,:tn,:fp,:fn,:id_modelo)")

#             # Define the parameters
#             params = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'id_modelo': id_model}

#             # Execute the query
#             conn.execute(query, params)
            
#             conn.commit()
#             return True
            
#     except SQLAlchemyError as e:
#         print(f"An error occurred while storing the model {model_id}.")
#         print(str(e))
#         return False
    
def query_showdata_head(id):

    # Create a connection to the PostgreSQL database
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

    # Get a Connection object from the Engine
    with engine.connect() as connection:
        query = sql.text(f"""
        SELECT "Father's qualification", "Mother's occupation", "Father's occupation", "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", "International", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)", "Target", "Marital status", "Application mode", "Application order", "Course", "Daytime/evening attendance", "Previous qualification", "Nacionality", "Mother's qualification", "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"
        FROM dataset_atributos WHERE id_dataset = :id_dataset
        LIMIT 5
    """)
        result = connection.execute(query, {'id_dataset': id})
        columns = result.keys()
        data = result.fetchall()
        return data,columns

    
def deactivate_all_models():
    """
    Deactivate all models in the 'models' table in a PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.

    Returns:
    - success (bool): True if the models were deactivated successfully, False otherwise.
    """
    try:
        engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

        # Store the user in the database
        with engine.connect() as connection:

            # Update the value in the table
            query = sql.text(f"""
                        UPDATE modelo SET 
                        is_active = FALSE
                        """)
            
            connection.execute(query)
            connection.commit()
            return True 

    except SQLAlchemyError as e:
        print(f"An error occurred while deactivating all models.")
        print(str(e))
        return False

    

def set_active_model(id_modelo):
    """
    Set the active model in the 'models' table in a PostgreSQL database.

    Parameters:
    - id_modelo (int): The ID of the active model.

    Returns:
    - success (bool): True if the active model was set successfully, False otherwise.
    """

    if deactivate_all_models():

        try:
            engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

            with engine.connect() as connection:

                # Update the value in the table
                query = sql.text(f"""
                            UPDATE modelo SET 
                            is_active = TRUE
                            WHERE id_modelo = :id_modelo
                            """)
                
                params = {'id_modelo': id_modelo}
    
                connection.execute(query, params)
                connection.commit()

                return True 

        except SQLAlchemyError as e:
            print(f"An error occurred while setting the active model.")
            print(str(e))
            return False 
    else:
        return False
    
    
    
def retrieve_model(id_modelo):
    """
    retrieves a model from a PostgreSQL database and saved file.

    Parameters:
    - id_modelo (int): The ID of the model to retrieve.

    Returns:
    - model (object): The model to retrieve.

    Raises:
    - SQLAlchemyError: If an error occurs while retrieving the dataset.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

        # Define the SQL query to select rows from the model table
        query = sql.text("""
        SELECT *
        FROM modelo
        WHERE id_modelo = :id_modelo
        """)

        params={"id_modelo": id_modelo}

        # retrieve the dataframe with the model info
        model_info = pd.read_sql(query, engine, params=params)

        # Check if the model_id exists in the table
        if not model_info.empty:
            # Get the model file name from the dataframe
            filename = model_info['filename'].values[0]
        else:
            print(f"Model with id {id_modelo} not found.")
            filename = None

        # Combine the folder path and the model file name
        directory="static/downloads"
        model_file_path = os.path.join(directory, filename)


        model = joblib.load(model_file_path)

        return model
    
    except SQLAlchemyError as e:
        print(f"An error occurred while retrieving the model {id_modelo} from model.")
        print(str(e))
        
        
def retrieve_active_model_info():
    """
    Retrieves the information about the active model from the 'model' table in the PostgreSQL database.

    Parameters:
    None

    Returns:
    - model_info (pd.DataFrame): A DataFrame that includes the rows where the 'is_active' column is True.
    """

    # Create a connection to the PostgreSQL database
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

    # Connect to the database
    with engine.connect() as conn:
        # Define the SQL query
        query = sql.text("SELECT * FROM modelo WHERE is_active = TRUE")

        # Execute the query and convert the result to a DataFrame
        model_info = pd.read_sql_query(query, engine)

    # Return the DataFrame
    return model_info

def get_evaluation(id_modelo):
    """
    Retrieves the evaluation results from the 'evaluation' table in the PostgreSQL database.

    Parameters:
    - id_modelo (int): The ID of the model.

    Returns:
    - evaluations (pd.DataFrame): A DataFrame that includes the rows where the 'id_model' column equals the provided model_id.
    """
    # Create a connection to the PostgreSQL database
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

    # Define your SQL query
    query = sql.text("SELECT tn, fp, fn, tp, f1_score, roc_auc, recall, precision, accuracy FROM avaliacao WHERE id_modelo = :id_modelo")
    # Execute the query and convert the result to a DataFrame
    evaluations = pd.read_sql_query(query, engine, params={"id_modelo": int(id_modelo)})

    return evaluations
    
    
def retrieve_model_info_dataf(table,id_modelo):
    """
    Retrieves information about a specific model from a database table.

    Parameters:
    table (str): The name of the table in the database.
    id_modelo (int): The ID of the model.

    Returns:
    list: A list of tuples where each tuple represents a row in the result set.
    """

    # Create a connection to the PostgreSQL database
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

    # Connect to the database
    with engine.connect() as conn:
        # Define the SQL query
        query = sql.text(f"SELECT * FROM {table} WHERE id_modelo = {id_modelo}")

        df = pd.read_sql_query(query, engine, params={"id_modelo": id_modelo})

    # Return the data
    # Convert the result to a DataFrame
    # Return the DataFrame
    return df

def export_dataset(id_dataset,tipo):
    """
    Exports a dataset from the 'dataset_atributos' table in the PostgreSQL database to a CSV file.

    Parameters:
    - id_dataset (int): The ID of the dataset to export.

    Returns:
    - success (bool): True if the dataset was exported successfully, False otherwise.
    """
    try:
        # Create a connection to the PostgreSQL database
        engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')

        # Connect to the database
        with engine.connect() as conn:
            # Define the SQL query
            if(tipo=="2"):
                query = sql.text(f"""
                SELECT "Father's qualification", "Mother's occupation", "Father's occupation", "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", "International", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)", "Marital status", "Application mode", "Application order", "Course", "Daytime/evening attendance", "Previous qualification", "Nacionality", "Mother's qualification", "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"
                FROM dataset_atributos
                WHERE id_dataset = :id_dataset
                """)
            else:
                query = sql.text(f"""
                SELECT "Father's qualification", "Mother's occupation", "Father's occupation", "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", "International", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)", "Target", "Marital status", "Application mode", "Application order", "Course", "Daytime/evening attendance", "Previous qualification", "Nacionality", "Mother's qualification", "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"
                FROM dataset_atributos
                WHERE id_dataset = :id_dataset
                """)
            # Execute the query and convert the result to a DataFrame
            df = pd.read_sql_query(query, engine, params={"id_dataset": id_dataset})

            # Define the filename
            filename = f"dataset.csv"

            # Define the filepath
            filepath = os.path.join("static/downloads_data", filename)

            # Export the DataFrame to a CSV file
            df.to_csv(filepath, index=False)

            return True

    except SQLAlchemyError as e:
        print(f"An error occurred while exporting the dataset {id_dataset}.")
        print(str(e))
        return False
    
    
def retrieve_dataset_info_type(tipo):
    """
    Retrieves information about the datasets from the 'dataset' table in the PostgreSQL database.

    Parameters:
    - tipo (str): The type of dataset to retrieve information for.

    Returns:
    - columns (list): A list of column names from the 'dataset' table.
    - data (list): A list of rows from the 'dataset' table.
    """
    # Create a connection to the PostgreSQL database
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
    # Define your SQL query
        query = sql.text(f"SELECT * FROM dataset WHERE tipo = '{tipo}'")

        result = conn.execute(query)
        columns = result.keys()
        data = result.fetchall()
        return columns, data
    
    
def store_evaluation(id_model,accuracy,precision,recall,roc_auc,f1_score,tn,fp,fn,tp):
    
    
    
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    
    with engine.connect() as conn:
        try:
            query = sql.text("INSERT INTO avaliacao (id_modelo, accuracy, precision, recall, roc_auc, f1_score, tn, fp, fn, tp) VALUES (:id_model, :accuracy, :precision, :recall, :roc_auc, :f1_score, :tn, :fp, :fn, :tp)")
            params = {'id_model': id_model, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc, 'f1_score': f1_score, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
            result = conn.execute(query, params)
            conn.commit()
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
        

def store_dataset_pred_read(df,name,tipo):
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        try:            
            query = sql.text("INSERT INTO dataset (name, tipo) VALUES (:name, :tipo) RETURNING id_dataset")
            params = {'name': name, 'tipo': tipo}
            result = conn.execute(query, params)
            id = result.fetchone()[0]
            
            # Add a new column to the DataFrame with the ID
            df['id_dataset'] = id
            
            # Insert the DataFrame into the 'dataset_atributos' table
            df.to_sql('dataset_atributos', conn, if_exists='append', index=False)
            
            conn.commit()
            
            return id
        except Exception as e:
            print(f"An error occurred: {e}")
            