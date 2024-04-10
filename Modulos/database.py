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
        print(data[0][0])
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
    
    
