import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io

BASE_DIR = r"C:\Users\WIN11\Desktop\FinalProject"

@st.cache_resource
def load_assets():
    try:
        p = joblib.load(os.path.join(BASE_DIR, "full_pipeline.pkl"))
        m = joblib.load(os.path.join(BASE_DIR, "model_metrics.pkl"))
        return p, m
    except: return None, None

def get_adaptive_logic(score, threshold, level):
    is_university = (level == "University")
    
    if score > threshold + 15:
        status = "Critical"
        if is_university:
            rec = "🚨 University Protocol: Referral to Academic Board, review cumulative GPA, and place under Academic Probation."
        else:
            rec = "🚨 School Protocol: Immediate parent conference, intensive remedial plan, and referral to Social Counselor."
            
    elif score > threshold:
        status = "At Risk"
        if is_university:
            rec = "⚠️ University Protocol: Notify via student email, recommend Office Hours, and guide to Learning Support Centers."
        else:
            rec = "⚠️ School Protocol: Parent notification, additional tutoring sessions, and weekly monitoring by Class Teacher."
            
    else:
        status = "Stable"
        if is_university:
            rec = "✅ University Status: Performance within safe range. Encourage continued academic engagement."
        else:
            rec = "✅ School Status: Good performance. Encourage participation in classroom activities."
            
    return status, rec

st.set_page_config(page_title="Academic DSS Dashboard", layout="wide")
pipeline, metrics = load_assets()

if not pipeline:
    st.error("⚠️ Model files not found. Please run model.py first.")
    st.stop()

# Sidebar Configuration
st.sidebar.header("⚙️ Control Panel")
risk_threshold = st.sidebar.slider("Risk Threshold (%)", 40, 90, 60)
academic_level = st.sidebar.radio("🏫 Select Analysis Environment:", ["School", "University"], horizontal=True)

st.title("🎓 Intelligent Academic Decision Support System")

# Performance & Explainability Sections
c_rep, c_shap = st.columns(2)
with c_rep:
    with st.expander("📄 Performance Metrics (Reproducible Results)"):
        st.dataframe(pd.DataFrame(metrics['Classification_Report']).transpose())

with c_shap:
    with st.expander("🔍 AI Interpretability (SHAP Analysis)"):
        f_names = [str(f).replace('num__', '').replace('cat__', '') for f in metrics['Proc_Feature_Names']]
        min_len = min(len(f_names), len(metrics['SHAP_Global']))
        shap_df = pd.DataFrame({'Feature': f_names[:min_len], 'Impact': metrics['SHAP_Global'][:min_len]}).sort_values(by='Impact', ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=shap_df, x='Impact', y='Feature', palette='mako', ax=ax)
        st.pyplot(fig)

# --- Individual Prediction Section ---
st.divider()
st.subheader(f"👤 Individual Prediction ({academic_level})")
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    with c1: age = st.number_input("Age", 15, 25, 18)
    with c2: study = st.slider("Study Time (1-4)", 1, 4, 2)
    with c3: fail = st.number_input("Past Failures", 0, 4, 0)
    with c4: abs_val = st.number_input("Absences", 0, 100, 5)
    
    if st.button("Run Prediction"):
        single_row = pd.DataFrame([[age, study, fail, abs_val]], columns=['age', 'studytime', 'failures', 'absences'])
        full_single = single_row.reindex(columns=metrics['Features']).fillna(0)
        for col in full_single.columns:
            if col in ['age', 'studytime', 'failures', 'absences']:
                full_single[col] = pd.to_numeric(full_single[col], errors='coerce').fillna(0)

        prob = pipeline.predict_proba(full_single)[:, metrics['Risk_Class_Index']][0] * 100
        st_label, st_rec = get_adaptive_logic(prob, risk_threshold, academic_level)
        st.info(f"Result: **{st_label}** | Risk Probability: **{prob:.2f}%**")
        st.write(f"**Recommendation:** {st_rec}")

# --- Batch Processing & Export ---
st.divider()
uploaded_file = st.file_uploader("Upload Student Records (CSV)", type="csv")
if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=';')
        subject_cols = [c for c in df_raw.columns if c.lower() in ['subject', 'course', 'class', 'school']]
        auto_subject = str(df_raw[subject_cols[0]].iloc[0]) if subject_cols else "Academic Record"
        st.success(f"📂 Record Detected: **{auto_subject}**")
        
        id_col = st.selectbox("👤 Select Student Identifier:", df_raw.columns)
        
        df_proc = df_raw.reindex(columns=metrics['Features'])
        for col in df_proc.columns:
            if df_proc[col].dtype == object: df_proc[col] = df_proc[col].fillna("Unknown")
            else: df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0)
        
        probs = pipeline.predict_proba(df_proc)[:, metrics['Risk_Class_Index']] * 100
        batch_results = [get_adaptive_logic(p, risk_threshold, academic_level) for p in probs]
        
        full_results = pd.DataFrame({
            'Student ID': df_raw[id_col], 'Risk Score (%)': probs.round(2),
            'Status': [r[0] for r in batch_results], 'Recommendation': [r[1] for r in batch_results]
        }).sort_values(by='Risk Score (%)', ascending=False)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Status Distribution")
            status_counts = full_results['Status'].value_counts()
            if len(status_counts) > 0:
                fig_p, ax_p = plt.subplots()
                status_counts.plot.pie(autopct='%1.1f%%', ax=ax_p)
                st.pyplot(fig_p)
        with c2:
            st.subheader("🔝 Top Critical Cases")
            st.bar_chart(full_results.head(5).set_index('Student ID')['Risk Score (%)'])

        st.subheader(f"📋 Analysis Report: {auto_subject}")
        st.dataframe(full_results, use_container_width=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            full_results.to_excel(writer, index=False, sheet_name='DSS_Analysis')
        st.download_button(f"📥 Download {auto_subject} Report", output.getvalue(), f"{auto_subject}_DSS_Report.xlsx")

    except Exception as e:
        st.error(f"❌ Error: {e}")