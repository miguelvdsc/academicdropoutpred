from flask import Flask, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user
import sqlalchemy

from Modulos.database import check_if_user_exists, check_pass, login_query, register_user

app = Flask(__name__)

app.config['SECRET_KEY'] = 'thisisasecretkey'


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id,name,tipo):
        self.id = id
        self.name = name
        self.tipo=tipo
        
@login_manager.user_loader
def user_loader(id):
    nome,tipo = login_query(id)
    user = User(id,nome,tipo)
    return user


    

@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        print (username,password)
        # Validate the username and password
        # This is where you should check them from your database
        idd = check_if_user_exists(username,password)
        if idd!=False and check_pass(idd,password):
            user=user_loader(idd)
            login_user(user)
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/handle_register', methods=['GET', 'POST'])
def handle_register():
    name = request.form['name']
    email= request.form['email']
    password = request.form['password']
    confirmpassword = request.form['confirmpassword']
    cargo = request.form['cargo']
    if(password==confirmpassword):
        if register_user(name,email,password,cargo):
            return redirect(url_for('login'))
    return redirect(url_for('register'))
    
        
    
@app.route('/register')
def register():
    return render_template('register.html')
# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('home'))

@app.route('/desativar')
def desativar():
    engine = sql.create_engine('postgresql://postgres:admin@localhost/frontend')
    with engine.connect() as conn:
        result = conn.execute("SELECT * FROM your_table")
        columns = result.keys()
        data = result.fetchall()
    return render_template('desativar.html', columns=columns, data=data)

if __name__ == '__main__':
    app.run()