USE postgres;
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    email VARCHAR(100),
    password VARCHAR(100)
    tipo VARCHAR(50)
);

CREATE TABLE dataset(
    id_dataset INT PRIMARY KEY,
)

CREATE TABLE atributos_aluno(
    id_dataset_linhas INT PRIMARY KEY,
    atributos_meter_aqui INT
    FOREIGN KEY (id_dataset_linhas) REFERENCES dataset(id_dataset)
)

CREATE TABLE alteracoes_dataset(
    id_alteracao INT PRIMARY KEY,
    id_dataset INT,
    FOREIGN KEY (id_dataset) REFERENCES dataset(id_dataset)
)

CREATE TABLE previsao(
    id_previsao INT PRIMARY KEY,
    id_versao INT FOREIGN KEY (id_versao) REFERENCES versao_modelo(id_versao),
    
)

CREATE TABLE previsao_aluno(
    id_previsao_aluno INT PRIMARY KEY,
    id_previsao INT FOREIGN KEY (id_previsao) REFERENCES previsao(id_previsao),
    abandono BOOLEAN,
)

CREATE TABLE modelo(
    id_modelo INT PRIMARY KEY,
    nome VARCHAR(50),
    algoritmo VARCHAR(50),
)

CREATE TABLE versao_modelo(
    id_versao INT PRIMARY KEY,
    id_modelo INT FOREIGN KEY (id_modelo) REFERENCES modelo(id_modelo),
    data timestamp,
    autor VARCHAR(50),
)

CREATE TABLE parametros(
    id_parametros INT PRIMARY KEY,
    id_versao INT FOREIGN KEY (id_versao) REFERENCES versao_modelo(id_versao),
    parametro VARCHAR(50),
    valor VARCHAR(50),
    tipo VARCHAR(50),
    id_versao INT FOREIGN KEY (id_versao) REFERENCES versao_modelo(id_versao),
)

CREATE TABLE avaliacao(
    id_avaliacao INT PRIMARY KEY,
    tp float,
    tn float,
    fp float,
    fn float,
    id_versao INT FOREIGN KEY (id_versao) REFERENCES versao_modelo(id_versao),
    f1 float,
    precision float,
    recall float,
    accuracy float,
)