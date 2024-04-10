import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Academic Dropout Prediction",
    page_icon="",
)


#st.sidebar.success("Select a demo above.")

# st.markdown(
#     """
#     Streamlit is an open-source app framework built specifically for
#     Machine Learning and Data Science projects.
#     **👈 Select a demo from the sidebar** to see some examples
#     of what Streamlit can do!
#     ### Want to learn more?
#     - Check out [streamlit.io](https://streamlit.io)
#     - Jump into our [documentation](https://docs.streamlit.io)
#     - Ask a question in our [community
#         forums](https://discuss.streamlit.io)
#     ### See more complex demos
#     - Use a neural net to [analyze the Udacity Self-driving Car Image
#         Dataset](https://github.com/streamlit/demo-self-driving)
#     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
# """
# )
import streamlit as st
import streamlit_authenticator as stauth
# Initialize connection.
conn = st.connection("postgresql", type="sql")

# Perform query.
df = conn.query('SELECT * FROM users;', ttl="0")
    
users = conn.query('SELECT * FROM users;', ttl="0")


credentials = {
    "usernames":{}
    
}

for _,user in users.iterrows():
    u={"email": user["tipo"],"name":user["name"], "password": user["password"]}
    credentials["usernames"][user["name"]] = u
    
authenticator = stauth.Authenticate(
    credentials,
    "name",
    "key"
)    

authenticator.login()


def gestao_dados():
    
    st.write("# Gestão de Dados")
    uploaded_file = st.file_uploader("Carregar Dataset")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
    

    if st.button("Selecionar Dataset",key="selecionar"):
        st.write("Selecionando Dataset...")
        
        conn = st.connection("postgresql", type="sql")

        # Perform query.
        df = conn.query('SELECT * FROM dataset;', ttl="0")
        
        if not df.empty:
            st.write(df)
            
        
        st.write("Dataset Selecionado")

def coiso():
    st.write("Coiso")
    



if st.session_state["authentication_status"]:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["username"]}*')
    st.write(st.session_state)
    # st.title('Some content')
    if st.session_state["name"]=='df':
        page_names_to_funcs = {
        "📁 Gestão de dados": gestao_dados,
        "Coiso": coiso,
        } 
        demo_name = st.sidebar.selectbox("", page_names_to_funcs.keys())
        page_names_to_funcs[demo_name]()
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')    
    
