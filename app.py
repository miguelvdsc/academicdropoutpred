from curses import flash
import json
from flask import Flask, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
import sqlalchemy

from Modulos.cleaning import translate_categorical_variables
from Modulos.database import change_password, check_if_user_exists, check_pass, insert_into_model, login_query, query_showdata_head, query_to_dataframe, register_user, retrieve_active_model_info, retrieve_model_info, select_from_table, select_from_table_dataset_type, select_from_table_id_one_dataset, select_head_dataset, set_active_model, store_dataset
from Modulos.database import modify_estado
from Modulos.model import train_model

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
            data_five,columns_five=query_showdata_head(id_dataset)
            lengc=len(columns_five)
            dd,ccc = select_from_table_id_one_dataset('dataset_atributos',id_dataset)
            lengd=len(dd)
            transdf=translate_categorical_variables(data_five,columns_five)
            print(columns_five)
            return render_template('show_dataset.html',user_type=current_user.tipo,data=data,columns=columns,data_five=transdf,columns_five=columns_five,lengc=lengc,lengd=lengd)
        return render_template('index.html',user_type=current_user.tipo)
    
@app.route('/create_model')
@login_required
def create_model():
    return render_template('create_model.html',user_type=current_user.tipo)

@app.route('/create_model_dataset', methods=['GET', 'POST'])
@login_required
def create_model_dataset():
    if request.method=="POST":
        if 'nome' in request.form:
            nome_modelo = request.form.get('nome')
        columns,data = select_from_table_dataset_type("treino")
        return render_template('create_model_dataset.html',user_type=current_user.tipo,data=data,columns=columns,nome_modelo=nome_modelo)

@app.route('/create_model_dataset_param', methods=['GET', 'POST'])
@login_required
def create_model_dataset_param():
    if request.method=="GET":
        return render_template('create_model_dataset_param.html',user_type=current_user.tipo)
    elif request.method=="POST":
        if 'nomem' in request.form:
            nome_modelo = request.form.get('nomem')
        if 'divi' in request.form:
            divi = request.form.get('divi')
        if 'selected_dataset' in request.form:
            selected_dataset = request.form.get('selected_dataset') 
           
        print(nome_modelo,divi,selected_dataset)
        return render_template('create_model_dataset_param.html',user_type=current_user.tipo,nome_modelo=nome_modelo,divi=divi,selected_dataset=selected_dataset)
        
@app.route('/model', methods=['GET', 'POST'])
@login_required
def model():
    if request.method=="POST":
        if 'nomem' in request.form:
            nome_modelo = request.form.get('nomem')
        if 'divi' in request.form:
            divi = request.form.get('divi')
        if 'selected_dataset' in request.form:
            selected_dataset = request.form.get('selected_dataset') 
        
        if 'criterion' in request.form:
            criterion = request.form.get('criterion')
        if 'splitter' in request.form:
            splitter = request.form.get('splitter')
        if 'max_depth' in request.form:
            max_depth = request.form.get('max_depth')
        if 'min_samples_split' in request.form:
            min_samples_split = request.form.get('min_samples_split')
        if 'min_samples_leaf' in request.form:
            min_samples_leaf = request.form.get('min_samples_leaf')
        if 'min_weight_fraction_leaf' in request.form:
            min_weight_fraction_leaf = request.form.get('min_weight_fraction_leaf')
        if 'max_features' in request.form:
            max_features = request.form.get('max_features')
        if 'random_state' in request.form:
            random_state = request.form.get('random_state')
        if 'max_leaf_nodes' in request.form:
            max_leaf_nodes = request.form.get('max_leaf_nodes')
        if 'min_impurity_decrease' in request.form:
            min_impurity_decrease = request.form.get('min_impurity_decrease')
        if 'class_weight' in request.form:
            class_weight = request.form.get('class_weight')
        if 'ccp_alpha' in request.form:
            ccp_alpha = request.form.get('ccp_alpha')
        if 'monotonic_cst' in request.form:
            monotonic_cst = request.form.get('monotonic_cst')
            
        form_data = {
            'criterion': str(criterion) if criterion in ['gini', 'entropy'] else 'gini',
            'splitter': str(splitter) if splitter in ['best', 'random'] else 'best',
            'max_depth': int(max_depth) if max_depth.isdigit() else None,
            'min_samples_split': int(min_samples_split) if min_samples_split.isdigit() else 2,
            'min_samples_leaf': int(min_samples_leaf) if min_samples_leaf.isdigit() else 1,
            'min_weight_fraction_leaf': float(min_weight_fraction_leaf) if min_weight_fraction_leaf.replace('.', '', 1).isdigit() else 0.0,
            'max_features': max_features if max_features in ['auto', 'sqrt', 'log2'] else None,
            'random_state': int(random_state) if random_state.isdigit() else None,
            'max_leaf_nodes': int(max_leaf_nodes) if max_leaf_nodes.isdigit() else None,
            'min_impurity_decrease': float(min_impurity_decrease) if min_impurity_decrease.replace('.', '', 1).isdigit() else 0.0,
            'class_weight': class_weight if class_weight in ['balanced', 'balanced_subsample'] else None,
            'ccp_alpha': float(ccp_alpha) if ccp_alpha.replace('.', '', 1).isdigit() else 0.0,
            'monotonic_cst': monotonic_cst if monotonic_cst else None,
        }
        df = query_to_dataframe('dataset_atributos', 'id_dataset', selected_dataset)
        print(df)
        split = float(divi) / 100.0
        id_model=insert_into_model(nome_modelo,form_data)
        if train_model(df,form_data,split,id_model):
            print('Modelo treinado com sucesso')
            if not (set_active_model(id_model)):
                flash('Erro ao ativar modelo', 'error')
            else:
                return render_template('model_info.html',user_type=current_user.tipo)
    elif request.method=="GET":
         return render_template('create_model_dataset_param.html',user_type=current_user.tipo)

        
@app.route('/model_info')
@login_required
def model_info():
    data = retrieve_active_model_info()
    parametros = data['parametros'].values[0]
    print(parametros)
    return render_template('model_info.html', user_type=current_user.tipo, data=parametros)
            
if __name__ == '__main__':
    app.run()