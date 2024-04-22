from flask import Flask, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
import sqlalchemy

from Modulos.database import change_password, check_if_user_exists, check_pass, login_query, register_user, select_from_table, select_from_table_id_one_dataset, select_head_dataset, store_dataset
from Modulos.database import modify_estado

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
    return render_template('index.html',user_type=current_user.tipo)


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
            return render_template('index.html',user_type=current_user.tipo)
    return render_template('login.html')

@app.route('/handle_register', methods=['GET', 'POST'])
@login_required
def handle_register():
    name = request.form['name']
    email= request.form['email']
    password = request.form['password']
    confirmpassword = request.form['confirmpassword']
    cargo = request.form['cargo']
    if(password==confirmpassword):
        if register_user(name,email,password,cargo):
            return render_template('login.html',user_type=current_user.tipo)
    return render_template('register.html',user_type=current_user.tipo)
    
        
    
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register')
@login_required
def register():
    return render_template('register.html',user_type=current_user.tipo)

@app.route('/desativar',methods=['GET', 'POST'])
@login_required
def desativar():
    if request.method == 'POST':
        selected_ids = request.args.getlist('user_id')
        for id in selected_ids:
            modify_estado(id)
        return render_template('desativar.html',user_type=current_user.tipo)
    elif request.method == 'GET':
        columns, data = select_from_table('users')
        return render_template('desativar.html', columns=columns, data=data,user_type=current_user.tipo)

@app.route('/changepw')
@login_required
def changepw():
    return render_template('changepw.html',user_type=current_user.tipo)
    
@app.route('/changepass', methods=['GET', 'POST'])
@login_required
def changepass():
    id = current_user.id
    password = request.form.get('password')
    confirmpassword = request.form.get('confirmpassword')
    if password and confirmpassword and password == confirmpassword:
        if change_password(id, password):
            return render_template('changepw.html',user_type=current_user.tipo)
        else:
            return 'An error occurred'
    

@app.route('/admin')
@login_required
def admin():
    return render_template('admin.html',user_type=current_user.tipo)

@app.route('/conta')
@login_required
def conta():
    return render_template('conta.html',user_type=current_user.tipo)

@app.route('/dados_inicial')
@login_required
def dados_inicial():
    return render_template('dados_inicial.html',user_type=current_user.tipo)

@app.route('/modelar')
@login_required
def modelar():
    return render_template('modelar.html',user_type=current_user.tipo)


@app.route('/dados_upload_file', methods=['GET', 'POST'])
@login_required
def dados_upload_file():
    if request.method=="GET":
        return render_template('dados_upload_file.html',user_type=current_user.tipo)
    elif request.method=="POST":
        file = request.files['fileda']
        name = request.form['nome']
        type = request.form['tipo']
        print(file,name,type)
        store_dataset(file,name,type)
        return render_template('dados_upload_file.html',user_type=current_user.tipo)
    
    
@app.route('/dados_select_dataset', methods=['GET'])
@login_required
def dados_select_dataset():
        columns, data = select_from_table('dataset')
        return render_template('dados_select_dataset.html', columns=columns, data=data,user_type=current_user.tipo)
    
@app.route('/show_dataset', methods=['GET', 'POST'])
@login_required
def show_dataset():
    if request.method=="POST":
        id_dataset = request.form['id_dataset']
        if id_dataset:
            data,columns = select_from_table_id_one_dataset('dataset',id_dataset)
            data_five,columns_five=select_head_dataset(id_dataset)
            lengc=len(columns_five)
            dd,ccc = select_from_table_id_one_dataset('dataset_atributos',id_dataset)
            lengd=len(dd)
            return render_template('show_dataset.html',user_type=current_user.tipo,data=data,columns=columns,data_five=data_five,columns_five=columns_five,lengc=lengc,lengd=lengd)
        return render_template('index.html',user_type=current_user.tipo)
if __name__ == '__main__':
    app.run()