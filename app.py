import pickle
import pandas as pd 
import numpy as np
import streamlit as st

st.set_page_config(page_title = 'Promotion Prediction',page_icon='🌀')
train_d = pd.read_csv('ProdigyTrain.csv')
train = train_d.drop(['employee_id','region'],axis = 1)
x_train = train.drop(['is_promoted'],axis = 1)
y_train = train['is_promoted']
num_cols = [col for col in x_train.columns if x_train[col].dtypes != 'O']   # ['no_of_trainings','age','previous_year_rating','length_of_service','KPIs_met >80%', 'awards_won?','avg_training_score']
cat_cols = [col for col in x_train.columns if x_train[col].dtypes == 'O']
model = pickle.load(open('ProdigyModel.sav','rb'))



hide_st_style = """
           <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

dep = list(train.department.unique())
edu = list(train.education.unique())
rec = list(train.recruitment_channel.unique())
head = st.container()
mid = st.container()
footer =  st.container()
st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Promotion Prediction 🏆</p>', unsafe_allow_html=True)
st.image('mainimage.jpg',width=700,caption='Based on Prodigy Dataset',use_column_width=True)
st.sidebar.header('Model Details')
options = st.sidebar.selectbox("Info",("Home 🏠","Visualize Data 📈", "Model Prediction 🎈"))
if options == 'Home 🏠':
    
    st.title('Sample Dataset')
    st.dataframe(train.head(20))
    st.download_button(label = '⬇️ Download Dataset',data='ProdigyTrain.csv' ,file_name= 'train_dataset.csv')
    st.markdown("_______________________________________________________________________________________________")
    st.title('Details of Project')
    st.write("This project is a Machine Learning and Data Analysis based. In this, we have the details of the employee like their education, department, score etc. [Dataset](https://www.kaggle.com/c/data-analytics-challenge-prodigy18/data) is derived from the [kaggle](https://www.kaggle.com) competition. The main goal of this project to predict that the employee will get promoted or not.")
    st.markdown("_______________________________________________________________________________________________")
    st.header("Links 🔗")
    st.markdown("[GitHub](https://github.com/hashGehlot03/)") 
    st.markdown("[Linkedin](https://www.linkedin.com/in/harish-gehlot-5338a021a)")
if options == 'Visualize Data 📈':
    st.title('Basic Details')
    st.write(f'Numerical Columns :-    {num_cols}')
    st.write(f'Categorical Columns :-  {cat_cols}')

    st.markdown("________________________________________________________________________________________________")
    st.title('Line plots')
    col_line = st.selectbox('Columns for plotting line',num_cols)
    st.title('Line Plot')
    st.line_chart(x_train[col_line].head(35),width = 80,height = 340)
    st.title('Area Plots')
    col_area = st.selectbox('Columns for plotting area',num_cols)
    st.title('Area Plot')
    st.area_chart(x_train[col_area].head(35),width=80,height = 340)
    st.title('Bubble Plot')
    col_bar = st.selectbox('Columns for plotting bubble',num_cols)
    st.title('Bar Plot')
    st.bar_chart(x_train[col_bar].head(35))

    st.markdown("_______________________________________________________________________________________________")
    st.header("Links 🔗")
    st.markdown("[Github](https://github.com/HashGehlot03)") 
    st.markdown("[Linkedin](https://www.linkedin.com/in/harish-gehlot-5338a021a)")
    

if options == 'Model Prediction 🎈':
    with st.form('Predict-Form'):
        department = st.selectbox('Department',dep)
        education = st.selectbox('Education',edu)
        gender = st.selectbox('Gender',['Male','Female'])
        if gender == 'Male':
            sex = 'm'
        else:
            sex = 'f'
        recruitment = st.selectbox('Recruitment Channel',rec)
        training = st.slider(label = 'No of trainings',min_value=0,max_value=10)
        age = st.text_input('Your Age')
        prev_rate = st.slider(label = 'Previous Year Rating',min_value=1,max_value=5)
        length_of_service = st.slider(label = 'Length Of Service',min_value=1,max_value=40,step = 2)
        kpi = st.checkbox(label = "KPI's met > 80% ?")
        Kpi = int(kpi)
        awards = st.checkbox(label = 'awards won any')
        Awards = int(awards)
        avg_training_score = st.slider(label = 'Average Training Score',min_value=40,max_value = 100,step = 2)
        predict = st.form_submit_button('Predict')
        st.markdown("_______________________________________________________________________________________________")
        if predict:
            pred = model.predict([[department,education,sex,recruitment,training,age,length_of_service,prev_rate,Kpi,Awards,avg_training_score]])
            if pred[0] == 0:
                st.write('Sorry  you are not promoted, Keep working Hard 😊')
            elif pred[0] == 1:
                st.write("Congrats You've got promotion  Party Time 🥳")

    

    st.markdown("_______________________________________________________________________________________________")
    st.header("Links 🔗")
    st.markdown("[Github](https://github.com/HashGehlot03)") 
    st.markdown("[Linkedin](https://www.linkedin.com/in/harish-gehlot-5338a021a)")

    


