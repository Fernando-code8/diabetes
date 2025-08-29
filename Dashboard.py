import streamlit as st
import pandas as pd
import numpy as np
#from dfply import *
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import spearmanr
import xgboost  as xgb
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve , auc, mean_squared_error, mean_absolute_error,recall_score,precision_score
import shap 
import joblib ## Salvar modelo como pickle
from streamlit_shap import st_shap

# Paineis iniciais
st.title("Dashboard - Predição de Risco de Diabetes 🩺")
st.markdown("Este painel apresenta uma análise exploratória e um modelo preditivo baseado no dataset Diabetes Health Indicators.")

# ==============================
# Carregar dados
# ==============================
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
df["id"] = df.index # criando um id de
df['situacao_entrevistado']=np.where(df["Diabetes_012"] == 0, "sem_diabetes", "pre_com_diabetes")
df=df.drop(columns=["Diabetes_012"])

aba1, aba2, aba3 = st.tabs(["Análise Exploratória", "Machine Learning (ML)", "Predição"])

with aba1:
    st.title("Análise Exploratória (EDA)")
    # ==============================
    # Distribuição das variáveis
    # ==============================
    
    df_selecao=df.drop(columns=["id"]).copy()
    variaveis=df_selecao.columns.tolist()
    var_escolhida = st.selectbox("Selecione a variável para visualizar:", variaveis)
    # Conta ocorrências da variável escolhida
    counts = df[var_escolhida].value_counts().reset_index()
    counts.columns = [var_escolhida, "count"]

    # Mostra gráfico
    st.subheader("Gráfico de frequência:")
    st.bar_chart(counts.set_index(var_escolhida))

    # ===========================================================
    #    Distribuição das variáveis com a variável alvo
    # ===========================================================
    st.subheader("Tabela cruzada com percentuais por linha da situação do entrevistado:")
    st.write(""" sem_diabetes = ausência de diabetes        
              pre_com_diabetes = indica pré-prediabetes e diabetes""")
    # Tabela cruzada com totais
    ct = pd.crosstab(df[var_escolhida], df["situacao_entrevistado"], margins=True, margins_name="Total")

    #Percentual por linha (sem incluir a linha Total)
    ct_percent = pd.crosstab(df[var_escolhida], df["situacao_entrevistado"],margins=True,margins_name="Total", normalize="index") * 100
    ct_percent = ct_percent.round(2)

    # Combinar totais e percentuais
    ct_combined = ct.copy()

    for col in df["situacao_entrevistado"].unique():
        ct_combined[f"{col} %"] = ct_percent[col]

    st.write(ct_combined)

     # ===========================================================
    #    Gráfico variáveis com a variável alvo
    # ===========================================================
    # Exemplo
    tab_cruz_sem_total = pd.crosstab(df[var_escolhida], df["situacao_entrevistado"])

    # Plot empilhado
    st.subheader("Gráfico de frequencia por situação do entrevistado:")
    st.bar_chart(tab_cruz_sem_total)

    # ==============================
    # Correlação
    # ==============================
    df_select=df_selecao.copy()
    df_select['situacao_entrevistado'] = df_select['situacao_entrevistado'].map({'pre_com_diabetes': 1, 'sem_diabetes': 0})
    # Correlação Spearman
    corr = df_select.corr(method='spearman')

    # Máscara para esconder metade superior da matriz
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(20, 15)) 
    sns.heatmap(corr, mask=mask, vmax=1.0, linewidths=0.01,square=True, annot=True, cmap='YlGnBu', linecolor='white',ax=ax)
    ax.set_title('Correlograma (Spearman)')
    st.subheader("Correlograma:")
    st.pyplot(fig)

with aba2:
    st.title("Machine Learning (ML)")
    # ==============================
    # Modelos
    # ==============================
    df_select=df.copy()
    df_select['situacao_entrevistado'] = df_select['situacao_entrevistado'].map({'pre_com_diabetes': 1, 'sem_diabetes': 0})


    # 4. Preparação dos dados

    # Amostragem

    #Variável de interesse
    alvo = df_select[['situacao_entrevistado']]
    # Remover a variável alvo do conjunto de dados
    df_select.drop(['situacao_entrevistado'],axis=1, inplace=True)


    # Separação em teste e treino

    ## Definindo treino e teste
    X_train, X_test, y_train, y_test=train_test_split(df_select,alvo, test_size=0.3, random_state=2025)

    ids_train = X_train["id"]  
    X_train = X_train.drop(columns=["id"])
    ids_test = X_test["id"]  
    X_test = X_test.drop(columns=["id"])

    # Logistic Regression
    log_reg = joblib.load("log_reg.pkl")

    # Random Forest
    rf = joblib.load("rf.joblib")

    # XGBoost
    mod_xgb = joblib.load("xgb.pkl")

    # ==============================
    # CURVA ROC
    # ==============================

    # Probabilidades da classe positiva (geralmente a segunda coluna [:,1])
    prob_xgb = mod_xgb.predict_proba(X_test)[:, 1]
    prob_log = log_reg.predict_proba(X_test)[:, 1]
    prob_fa = rf.predict_proba(X_test)[:, 1]

    # Calcular pontos da curva ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, prob_xgb)
    fpr_log, tpr_log, _ = roc_curve(y_test, prob_log)
    fpr_fa, tpr_fa, _ = roc_curve(y_test, prob_fa)

    # Calcular AUC
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    auc_log = auc(fpr_log, tpr_log)
    auc_fa = auc(fpr_fa, tpr_fa)

    # Plot da curva ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_xgb, tpr_xgb, label=f'XGB (AUC = {auc_xgb:.2f})')
    ax.plot(fpr_log, tpr_log, label=f'Logística (AUC = {auc_log:.2f})')
    ax.plot(fpr_fa, tpr_fa, label=f'Floresta Aleatória (AUC = {auc_fa:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)')
    ax.set_xlabel('FPR - Taxa de Falsos Positivos')
    ax.set_ylabel('TPR - Sensibilidade')
    ax.set_title('Curva ROC')
    ax.legend(loc='lower right')
    ax.grid()

    st.subheader("Comparação de modelos - Curva ROC:")
    st.pyplot(fig)

    st.subheader("Métricas de Sucesso da Regressão Logística:")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 AUC\n0.82")

    with col2:
        st.markdown("### ✅ Acurácia\n0.72")

    with col3:
        st.markdown("### ⚡ Recall\n0.77")



    # ==============================
    # Matriz de confusão
    # ==============================
    previsao_logistico = log_reg.predict(X_test)

    # Matriz de confusão
    labels = [0, 1]
    cm_logistico = confusion_matrix(y_test, previsao_logistico,labels=labels)

    # Matriz de confusão bonitinha

    # Logistica
    cm_logistico_df = pd.DataFrame(cm_logistico, index=[f'Real {label}' for label in labels],
                            columns=[f'Previsto {label}' for label in labels])
    cm_logistico_df['Total Linha'] = cm_logistico_df.sum(axis=1)
    cm_logistico_df.loc['Total Coluna'] = cm_logistico_df.sum()

    # Cálculo de métricas
    tn_log, fp_log, fn_log, tp_log = cm_logistico.ravel()
    especificidade_log = tn_log / (tn_log + fp_log)
    vpn_log = tn_log / (tn_log + fn_log)
    recall_log = recall_score(y_test, previsao_logistico)
    media_rec_spec_log = np.sqrt(recall_log * especificidade_log)


    st.subheader("Matriz de Confusão:")
    st.write(cm_logistico_df)

    # ==============================
    # Interpretação
    # ==============================
    # Shap
    
    sample_size = 50000  # ajustar conforme performance
    X_sample = X_train.sample(n=sample_size, random_state=42)
    # Objeto explainer - XGBoost:TreeExplainer, Floresta Aleatoria:Explainer, Regressão Logística: LinearExplainer
    explainer = shap.LinearExplainer(log_reg,X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_train)

    # Calcular os valores SHAP no conjunto de treino
    shap_values = explainer(X_train)

    
    st.write(""" # Interpretabilidade """)
    # Resumo Geral
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values)
    st.subheader("Gráfico de importância:")
    st.pyplot(fig)

    # Beeswarm
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    st.subheader("SHAP Beeswarm Plot:")
    st.pyplot(fig)

    # efeito 
    # Selecionar variável
    variaveis2 = X_train.columns.tolist()
    var_escolhida = st.selectbox("Selecione a variável para visualizar:", variaveis2)
    feature_index = X_train.columns.get_loc(var_escolhida)

    # Exibir gráfico interativo
    st.subheader(f"Gráfico de dependência SHAP: {var_escolhida}")
    st_shap(shap.plots.scatter(shap_values[:, feature_index]),
                height=600  # aumentar altura
                )

with aba3:
    st.title("Predição")
    # ==============================
    # Relatório
    # ==============================

    relatorio = pd.read_csv("relatorio.csv")
    amostra = relatorio.sample(10000, random_state=42)
    tab=relatorio.Grupo_de_Risco.value_counts()
    st.subheader("Frequência - Grupo de Risco:")
    st.markdown("""
    **Baixo Risco** = Menor que 50% de chance de ter diabetes  
    **Médio Risco** = De 50% a 70% de chance de ter diabetes  
    **Alto Risco** = Maior que 70% de chance de ter diabetes
    """)
    st.write(tab)

    st.subheader("Relatório de Predição de Risco de Diabetes:")
    st.dataframe(amostra)




    

    

   


    

    

    



