import gradio as gr
import sys
sys.path.append("../")
import pandas as pd
from utils.data_utils import DataLoader
import xgboost as xgb
import optuna
import lightgbm as lgb
from importlib import reload
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from utils import visualization
from utils.encoding import Encoded_Features, CatEncoderWrapper
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from io import StringIO
from interpret.blackbox import LimeTabular
from interpret import show
import lime
import lime.lime_tabular

reload(visualization)
reload(sys.modules['utils.data_utils'])
reload(sys.modules['utils.encoding'])

data_loader = DataLoader(file_path='../data/raw/Churn_Modelling.xls' , target_column='Exited', test_size=0.2, random_state=42, clean_data=True)

X_test, y_test = data_loader.get_test_data()
encoder = CatEncoderWrapper(columns=['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Tenure'], dtype='int64')
X_test = encoder.transform(X_test)

def plot_roc():
    model = joblib.load('../models/CatBoost/model_cb2.sav')
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return fig


col_list = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

def process_csv_text(temp_file):
    if isinstance(temp_file, str):
      df = pd.read_csv(StringIO(temp_file))
    else:
      df = pd.read_csv(temp_file.name)
    print(df)
    return df

def data_frame(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    encoder_model = joblib.load('../models/encoder_model.sav')
    credit_card_dict = {"Yes": 1, "No": 0}
    active_member_dict = {"Yes": 1, "No": 0}
    credit_card = credit_card_dict[HasCrCard]
    active_member = active_member_dict[IsActiveMember]
    list = [CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, credit_card, active_member, EstimatedSalary]
    df = encoder_model.transform(pd.DataFrame([list], columns=col_list))
    return df

def predict(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    df = data_frame(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)    
    model = joblib.load('../models/CatBoost/model_cb3.sav')
    return model.predict(df)[0]

def explain_plot(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    df = data_frame(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
    model = joblib.load('../models/CatBoost/model_cb3.sav')
    X_enc, y_enc = data_loader.get_data_enc()
    predict_fn = lambda x: model.predict_proba(x).astype(float)
    explainer = lime.lime_tabular.LimeTabularExplainer(class_names=model.classes_, 
                                                       training_data=X_enc.values,
                                                       feature_names=model.feature_names_,
                                                        mode='classification',
                                                        discretize_continuous=False)

    exp = explainer.explain_instance(data_row=df.iloc[0], predict_fn=predict_fn, num_features=10)
    fig = exp.as_pyplot_figure()
    return {data_tbl: pd.DataFrame(exp.as_list(), columns=['Feature', 'Weight']),
            plot: exp.as_pyplot_figure()}


with gr.Blocks() as demo:
    with gr.Tab("LIME"):
        with gr.Row():
            with gr.Column():
                credit_score = gr.Slider(300, 850, step=1, label="Credit Score")
                geography = gr.Dropdown(["France", "Spain", "Germany"], label="Country")
                gender = gr.Radio(['Male', 'Female'], label="Gender")
                age = gr.Number(label="Age")
                tenure = gr.Slider(0, 10, step=1, label="Tenure")
                balance = gr.Number(label="Balance")
                num_products = gr.Slider(1, 4, step=1, label="Number of Products")
                credit_card = gr.Radio(["Yes", "No"], label="Has Credit Card")
                active_member = gr.Radio(["Yes", "No"], label="Is Active Member")
                estimated_salary = gr.Number(label="Estimated Salary")
            with gr.Column():
                output = gr.Textbox(label="Prediction")
                data_tbl = gr.Dataframe(headers=['Feature', 'Weight'], datatype=['str', 'number'], row_count=10, col_count=(2, "fixed"), interactive=True)
                plot = gr.Plot()
                submit_btn = gr.Button("Predict")
                exp_btn = gr.Button("Explain")
        exp_btn.click(fn=explain_plot, inputs= [credit_score, geography, gender, age, tenure, balance, num_products, credit_card, active_member, estimated_salary], outputs=[data_tbl, plot], api_name="Explain Churn Lime")
        submit_btn.click(fn=predict, inputs= [credit_score, geography, gender, age, tenure, balance, num_products, credit_card, active_member, estimated_salary], outputs=output, api_name="Predict Churn Lime")

    with gr.Tab("SHAPLEY"):
        with gr.Row():
            with gr.Column():
                upload_btn = gr.UploadButton(label="Upload CSV", file_types=["csv"])
                data = gr.Dataframe(headers=col_list,
                datatype=["number", "str", "str", "number", "number", "number", "number", "number", "number", "number"],
                row_count=5,
                col_count=(len(col_list), "fixed"), interactive=True)
            with gr.Column():
                output = gr.Textbox(label="Prediction")
                submit_btn = gr.Button("Predict")
        upload_btn.upload(fn=process_csv_text, inputs=upload_btn, outputs=data, api_name="Upload CSV")
        submit_btn.click(fn=predict, inputs= [credit_score, geography, gender, age, tenure, balance, num_products, credit_card, active_member, estimated_salary], outputs=output, api_name="Predict Churn Shapley")
    
    with gr.Tab("COUNTERFACTUAL"):
        with gr.Row():
            with gr.Column():
                credit_score = gr.Slider(300, 850, step=1, label="Credit Score")
                geography = gr.Dropdown(["France", "Spain", "Germany"], label="Country")
                gender = gr.Radio(['Male', 'Female'], label="Gender")
                age = gr.Number(label="Age")
                tenure = gr.Slider(0, 10, step=1, label="Tenure")
                balance = gr.Number(label="Balance")
                num_products = gr.Slider(1, 4, step=1, label="Number of Products")
                credit_card = gr.Radio(["Yes", "No"], label="Has Credit Card")
                active_member = gr.Radio(["Yes", "No"], label="Is Active Member")
                estimated_salary = gr.Number(label="Estimated Salary")
            with gr.Column():
                output = gr.Textbox(label="Prediction")
                submit_btn = gr.Button("Predict")
        submit_btn.click(fn=predict, inputs= [credit_score, geography, gender, age, tenure, balance, num_products, credit_card, active_member, estimated_salary], outputs=output, api_name="Predict Churn Counterfactual")



demo.launch()