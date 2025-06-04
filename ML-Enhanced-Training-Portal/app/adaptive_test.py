import streamlit as st
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import os
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import altair as alt

# Custom SAC Agent Class (simplified version for inference only)
@tf.keras.utils.register_keras_serializable()
class SACAgent:
    def __init__(self, actor_model):
        self.actor = actor_model

    def act(self, state):
        state_actor = np.reshape(state, (1, 1, len(state)))
        action_prob = self.actor(state_actor, training=False)
        return np.argmax(action_prob.numpy()[0])

    def get_config(self):
        return {}

# ✅ Load SAC AI Models Safely
try:
    model_path = "../model/sac_adaptive_learning_model.h5"
    actor_model = load_model(f"{model_path}_actor.keras")
    model = SACAgent(actor_model)
    st.sidebar.success("✅ SAC model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading the SAC model: {str(e)}")
    model = None

# ✅ Load Questions from JSON
try:
    questions_path = os.path.join("..", "data", "questions.json")
    with open(questions_path, "r") as f:
        questions_data = json.load(f)
except Exception as e:
    st.error(f"⚠️ Error loading questions.json: {str(e)}")
    questions_data = {}

# ✅ Define Module Encoding
module_map = {"Easy": 0, "Medium": 1, "Hard": 2}
reverse_module_map = {0: "Easy", 1: "Medium", 2: "Hard"}

# ✅ Initialize Session State Variables
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = None
if "test_history" not in st.session_state:
    st.session_state.test_history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "domain_module_progress" not in st.session_state:
    st.session_state.domain_module_progress = {}
if "test_in_progress" not in st.session_state:
    st.session_state.test_in_progress = False
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "student_id" not in st.session_state:
    st.session_state.student_id = None
if "performance_data" not in st.session_state:
    st.session_state.performance_data = []
if "student_roles" not in st.session_state:
    st.session_state.student_roles = {}
if "edge_question_added" not in st.session_state:
    st.session_state.edge_question_added = False
if "job_preference" not in st.session_state:
    st.session_state.job_preference = ""

if "edge_question_tracker" not in st.session_state:
    st.session_state.edge_question_tracker = {}

def save_performance_data(data):
    filename = f"student_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

# ✅ UI Header
st.title("📖 Adaptive Learning Quiz with SAC AI")
st.write("Select a domain and answer **5 questions**. The SAC AI will recommend the next module based on your performance.")

# ✅ Step 1: Domain Selection, Student ID, and Job Preference Input
if not st.session_state.test_in_progress and st.session_state.selected_domain is None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧑‍🎓 Enter Student Details:")
        student_id_input = st.text_input("Student ID:", key="student_id_input")
        if student_id_input and student_id_input in st.session_state.student_roles:
            stored_job_pref = st.session_state.student_roles[student_id_input].get("job_preference", "")
            job_preference_input = st.text_input("What is your job preference?", value=stored_job_pref, key="job_preference_input")
            st.info("Welcome back! Your previous job preference has been loaded.")
        else:
            job_preference_input = st.text_input("What is your job preference?", key="job_preference_input")

    with col2:
        st.subheader("📚 Select Your Domain:")
        available_domains = list(questions_data.keys()) if questions_data else []
        domain = st.selectbox("Choose a domain:", [""] + available_domains)

    if domain and student_id_input and st.button("Start Test", type="primary"):
        st.session_state.student_id = student_id_input
        st.session_state.job_preference = job_preference_input
        st.session_state.selected_domain = domain
        st.session_state.test_in_progress = True
        st.session_state.current_question = 0
        st.session_state.user_answers = {}
        st.session_state.start_time = time.time()
        st.session_state.edge_question_added = False

        if student_id_input in st.session_state.student_roles:
            prev_module = st.session_state.student_roles[student_id_input].get("recommended_module", "Easy")
            st.session_state.domain_module_progress[domain] = prev_module
        else:
            if domain not in st.session_state.domain_module_progress:
                st.session_state.domain_module_progress[domain] = "Easy"

        st.session_state.current_module = st.session_state.domain_module_progress[domain]
        st.rerun()

# ✅ Step 2: Quiz Interface
if st.session_state.selected_domain and st.session_state.test_in_progress:
    if st.session_state.start_time:
        elapsed_time = int(time.time() - st.session_state.start_time)
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        st.info(f"⏱️ Time Elapsed: {minutes:02d}:{seconds:02d}")

    st.subheader(f"📚 Domain: {st.session_state.selected_domain} | Level: {st.session_state.current_module}")
    st.write(f"Student ID: {st.session_state.student_id}")

    try:
        domain_data = questions_data.get(st.session_state.selected_domain, {})
        current_questions = domain_data.get(st.session_state.current_module, [])[:5]

        student_domain_key = f"{st.session_state.student_id}_{st.session_state.selected_domain}"
        if (st.session_state.student_id in st.session_state.student_roles
            and not st.session_state.edge_question_added
            and student_domain_key not in st.session_state.edge_question_tracker):

            edge_questions = domain_data.get("Edge", [])
            if edge_questions:
                current_questions.append(edge_questions[0])
                st.session_state.edge_question_added = True
                st.session_state.edge_question_tracker[student_domain_key] = True

        if not current_questions:
            st.error(f"⚠️ No questions available for {st.session_state.current_module} level.")
            st.session_state.test_in_progress = False
            st.session_state.selected_domain = None
            st.rerun()

        if st.session_state.current_question < len(current_questions):
            question_data = current_questions[st.session_state.current_question]

            with st.container():
                if question_data.get("edge_case", False):
                    st.markdown("### ⚠️ Edge Case Question:")
                st.write(f"**Q{st.session_state.current_question + 1}:** {question_data['question']}")

                selected_answer = st.radio(
                    "Select your answer:",
                    question_data["options"],
                    key=f"q{st.session_state.current_question}",
                    index=None
                )

                if st.button("Submit Answer", type="primary"):
                    if selected_answer is not None:
                        st.session_state.user_answers[st.session_state.current_question] = {
                            "question": question_data["question"],
                            "selected": selected_answer,
                            "correct": question_data["answer"],
                            "is_correct": selected_answer == question_data["answer"]
                        }
                        if selected_answer == question_data["answer"]:
                            st.success("✅ Correct!")
                        else:
                            st.error(f"❌ Incorrect. The correct answer is: {question_data['answer']}")

                        if st.session_state.current_question + 1 < len(current_questions):
                            st.session_state.current_question += 1
                            st.rerun()
                        else:
                            st.session_state.show_results = True
                            st.session_state.test_in_progress = False
                            st.rerun()
    except Exception as e:
        st.error(f"⚠️ Error displaying questions: {str(e)}")
        st.session_state.test_in_progress = False
        st.rerun()

# ✅ Step 3: Show Results
if st.session_state.show_results:
    try:
        correct_answers = sum(1 for ans in st.session_state.user_answers.values() if ans["is_correct"])
        score_percentage = (correct_answers / len(st.session_state.user_answers)) * 100
        time_spent = round((time.time() - st.session_state.start_time) / 60, 2)

        st.success(f"🎯 Your Score: **{score_percentage:.2f}%**")
        st.info(f"⏱️ Time Spent: **{time_spent:.2f}** minutes")

        if model is not None:
            features = np.array([
                score_percentage,
                time_spent,
                len(st.session_state.test_history) + 1,
                module_map[st.session_state.current_module]
            ])
            try:
                predicted_module_index = model.act(features)
                recommended_module = reverse_module_map[predicted_module_index]

                current_level = st.session_state.current_module
                if score_percentage < 50:
                    recommended_module = "Medium" if current_level == "Hard" else "Easy"
                elif 50 <= score_percentage <= 85:
                    recommended_module = "Medium" if current_level == "Easy" else "Hard"
                elif score_percentage > 85:
                    recommended_module = "Hard"

                st.success(f"🤖 SAC AI Recommended Next Module: **{recommended_module}**")
            except Exception as e:
                st.error(f"⚠️ Model prediction error: {str(e)}")
                recommended_module = st.session_state.current_module
        else:
            recommended_module = st.session_state.current_module
            st.warning("⚠️ AI model not loaded. Using default recommendation.")

        st.session_state.domain_module_progress[st.session_state.selected_domain] = recommended_module
        st.session_state.student_roles[st.session_state.student_id] = {
            "job_preference": st.session_state.job_preference,
            "recommended_module": recommended_module
        }

        performance_record = {
            "Student_ID": st.session_state.student_id,
            "Domain": st.session_state.selected_domain,
            "Previous_Score": score_percentage,
            "Time_Spent": time_spent,
            "Attempt_Count": len(st.session_state.test_history) + 1,
            "Module_Difficulty": st.session_state.current_module,
            "Next_Recommended_Module": recommended_module,
            "Job_Preference": st.session_state.job_preference
        }
        st.session_state.performance_data.append(performance_record)

        test_summary = {
            "Student_ID": st.session_state.student_id,
            "Domain": st.session_state.selected_domain,
            "Score": f"{score_percentage:.2f}%",
            "Time_Spent": f"{time_spent:.2f} min",
            "Previous Module": st.session_state.current_module,
            "Next Module": recommended_module,
            "Job_Preference": st.session_state.job_preference
        }
        st.session_state.test_history.append(test_summary)

        st.subheader("📊 Test History")
        history_df = pd.DataFrame(st.session_state.test_history)
        st.dataframe(history_df)

        if st.session_state.performance_data:
            csv_data = save_performance_data(st.session_state.performance_data)
            st.download_button(
                label="📥 Download Performance Data",
                data=csv_data,
                file_name="student_performance_data.csv",
                mime="text/csv"
            )

        # ✅ Student Performance Dashboard
        st.subheader("📈 Student Domain Performance")
        perf_df = pd.DataFrame(st.session_state.performance_data)
        if not perf_df.empty:
            domain_avg = perf_df.groupby("Domain")["Previous_Score"].mean().reset_index()
            chart = alt.Chart(domain_avg).mark_bar(color="#4e79a7").encode(
                x=alt.X("Domain:N", sort="-y"),
                y=alt.Y("Previous_Score:Q", title="Average Score (%)"),
                tooltip=["Domain", "Previous_Score"]
            ).properties(title="Average Performance by Domain", width=600)
            st.altair_chart(chart, use_container_width=True)

            # Additional: Student's average performance
            student_perf = perf_df[perf_df["Student_ID"] == st.session_state.student_id]
            if not student_perf.empty:
                student_avg = student_perf.groupby("Domain")["Previous_Score"].mean().reset_index()
                st.subheader("📌 Your Performance by Domain")
                chart2 = alt.Chart(student_avg).mark_bar(color="#f28e2b").encode(
                    x=alt.X("Domain:N", sort="-y"),
                    y=alt.Y("Previous_Score:Q", title="Your Avg Score (%)"),
                    tooltip=["Domain", "Previous_Score"]
                ).properties(title="Your Domain-wise Performance", width=600)
                st.altair_chart(chart2, use_container_width=True)

        if st.button("Start New Test", type="primary"):
            st.session_state.current_module = recommended_module
            st.session_state.selected_domain = None
            st.session_state.current_question = 0
            st.session_state.user_answers = {}
            st.session_state.show_results = False
            st.session_state.start_time = None
            st.rerun()
    except Exception as e:
        st.error(f"⚠️ Error displaying results: {str(e)}")
        st.exception(e)

# Sidebar Info
with st.sidebar:
    st.subheader("🤖 Model Information")
    st.write("Using SAC (Soft Actor-Critic) model for adaptive learning.")
    st.info("This model uses an LSTM-based actor to predict the optimal next module based on student performance metrics.")

    if model:
        st.success("Model Status: Loaded ✅")
        if st.checkbox("Show Model Debug Info"):
            st.write("Model input shape for actor:", "(batch_size, time_steps, features) = (1, 1, 4)")
            test_features = np.array([75.0, 5.0, 1, 1])
            st.write("Test features:", test_features)
            try:
                reshaped_input = np.reshape(test_features, (1, 1, 4))
                st.write("Reshaped input shape:", reshaped_input.shape)
            except Exception as e:
                st.error(f"Reshaping error: {str(e)}")
    else:
        st.error("Model Status: Not Loaded ❌")

    st.divider()
    st.write("Developed by Adaptive Learning Team")