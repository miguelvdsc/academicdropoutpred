import pandas as pd
import sqlalchemy as sql
import bcrypt

def login_query(id):
    
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        query = sql.text(f"SELECT name, tipo FROM users WHERE id = {id}")
        result = conn.execute(query)
        data = result.fetchall()
        name=data[0][0]
        tipo=data[0][1]
        return name,tipo
    
    
def check_if_user_exists(username,password):
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

def check_pass(id,password):
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
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        query = sql.text("UPDATE users SET estado = 0 WHERE id = :id")
        params = {'id': id}
        result = conn.execute(query, params)
        conn.commit()
        
        
def select_from_table(table):
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