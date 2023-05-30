import sys

import gradio as gr

sys.path.append("../")
from importlib import reload
from io import StringIO

import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import (auc, roc_curve)
from utils import visualization
from utils.data_utils import DataLoader
from utils.encoding import CatEncoderWrapper
import dice_ml

reload(visualization)
reload(sys.modules['utils.data_utils'])
reload(sys.modules['utils.encoding'])

data_loader = DataLoader(file_path='../data/raw/Churn_Modelling.xls' , target_column='Exited', test_size=0.2, random_state=42, clean_data=True)
model = joblib.load('../models/CatBoost/model_cb3.sav')
X_test, y_test = data_loader.get_test_data()
encoder = CatEncoderWrapper(columns=['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Tenure'], dtype='int64')
X_test = encoder.transform(X_test)


col_list = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
col_list_enc = ['Geography_France', 'Geography_Germany', 'Geography_Spain', 'Gender_Female', 'Gender_Male', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']

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
    pred = model.predict_proba(df)[0]
    return {"Not Churned": float(pred[0]), "Churned": float(pred[1])} # model.predict(df)[0]

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
    exp.as_pyplot_figure()
    return {data_tbl: pd.DataFrame(exp.as_list(), columns=['Feature', 'Weight']),
            plot: exp.as_pyplot_figure()}



def data_shap(df):
    encoder_model = joblib.load('../models/encoder_model.sav')
    df = encoder_model.transform(df)
    return df

def predict_shap(df):
    df = data_shap(df)
    model = joblib.load('../models/CatBoost/model_cb3.sav')
    return pd.DataFrame(model.predict(df), columns=['Prediction'])

def shap_plot(df):
    df = data_shap(df)
    model = joblib.load('../models/CatBoost/model_cb3.sav')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    shap.summary_plot(shap_values,df, show=False)
    plt.savefig('shap.png', bbox_inches='tight')
    return 'shap.png'

def cf_explain(credit_score, geography, gender, age, tenure, balance, num_products, credit_card, active_member, estimated_salary, check_box):
    data = data_frame(credit_score, geography, gender, age, tenure, balance, num_products, credit_card, active_member, estimated_salary)
    X_enc, y_enc = data_loader.get_data_enc()
    model = joblib.load('../models/CatBoost/model_cb3.sav')
    df = pd.concat([X_enc, y_enc], axis=1)
    df['NumOfProducts'] = df['NumOfProducts'].astype(float)
    df['IsActiveMember'] = df['IsActiveMember'].astype(float)

    data_dice = dice_ml.Data(dataframe=df, continuous_features=['Age', 'NumOfProducts', 'EstimatedSalary', 'IsActiveMember'], outcome_name='Exited')
    model_dice = dice_ml.Model(model=model, backend="sklearn")
    explainer = dice_ml.Dice(data_dice, model_dice, method="random")

    input_datapoint = data
    features_to_vary= check_box

    permitted_range={'Age':[20,30], 'EstimatedSalary':[116000, 119000]}

    cf = explainer.generate_counterfactuals(input_datapoint, total_CFs=20, desired_class="opposite", permitted_range=permitted_range, features_to_vary=features_to_vary)
    return cf.cf_examples_list[0].final_cfs_df



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
                output = gr.Label("Churn Prediction")
                data_tbl = gr.Dataframe(headers=['Feature', 'Weight'], datatype=['str', 'number'], row_count=10, col_count=(2, "fixed"), interactive=True)
                plot = gr.Plot()
                submit_btn = gr.Button("Predict")
                exp_btn = gr.Button("Explain")
        exp_btn.click(fn=explain_plot, inputs= [credit_score, geography, gender, age, tenure, balance, num_products, credit_card, active_member, estimated_salary], outputs=[data_tbl, plot], api_name="Explain Churn Lime")
        submit_btn.click(fn=predict, inputs= [credit_score, geography, gender, age, tenure, balance, num_products, credit_card, active_member, estimated_salary], outputs=output, api_name="Predict Churn Lime")

    with gr.Tab("SHAPLEY"):
        with gr.Row():
            with gr.Column():
                data = gr.Dataframe(headers=col_list,
                datatype=["number", "str", "str", "number", "number", "number", "number", "number", "number", "number"],
                row_count=1,
                col_count=(len(col_list), "fixed"), interactive=True)
                upload_btn = gr.UploadButton(label="Upload CSV", file_types=["csv"])
            with gr.Column():
                output = gr.DataFrame(headers=['Prediction'], datatype=['number'], row_count=1, col_count=(1, "fixed"), interactive=True)
                plot_shap_val = gr.Image()
                explain_btn = gr.Button("Explain")
                submit_btn = gr.Button("Predict")
        upload_btn.upload(fn=process_csv_text, inputs=upload_btn, outputs=data, api_name="Upload CSV")
        submit_btn.click(fn=predict_shap, inputs= data, outputs=output, api_name="Predict Churn Shapley")
        explain_btn.click(fn=shap_plot, inputs=data, outputs=plot_shap_val, api_name="Explain Churn Shapley")
        
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
                output = gr.Label("Churn Prediction")
                submit_btn = gr.Button("Predict")
                explain_cf = gr.Button("Generate Counterfactual")
                with gr.Row():
                    check_box =  gr.CheckboxGroup(['Age', 'NumOfProducts', 'EstimatedSalary', 'IsActiveMember'], label="Features to vary")
                explain_df = gr.Dataframe(headers=col_list_enc,
                                          datatype=["number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number"],
                                          row_count=1, col_count=(len(col_list_enc), "fixed"), interactive=True)
        submit_btn.click(fn=predict, inputs= [credit_score, geography, gender, age, tenure, balance, num_products, credit_card, active_member, estimated_salary], outputs=output, api_name="Predict Churn Counterfactual")
        explain_cf.click(fn=cf_explain, inputs= [credit_score, geography, gender, age, tenure, balance, num_products, credit_card, active_member, estimated_salary, check_box], outputs=explain_df, api_name="Explain Churn Counterfactual")


demo.launch(share=True)