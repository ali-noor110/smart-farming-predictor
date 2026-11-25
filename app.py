import os
import sys
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
sys.path.insert(0, os.path.dirname(__file__))

from astar import astar_recommend
from genetic_algorithm import run_ga, run_ga_experiment
from random_forest import (
    MODEL_PATH,
    get_accuracy,
    get_evaluation,
    resolve_dataset_path,
    rf_predict,
    train_and_save,
)

st.set_page_config(
    page_title="Smart Farming Predictor",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #173c25, #2d7a48);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; margin: 0; }
    .hero p { color: #d7eddc; margin: 0.4rem 0 0; }
    .card {
        background: #ffffff;
        border: 1px solid #dce9df;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    resolved_dataset = resolve_dataset_path()
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Training Random Forest model"):
            train_and_save(resolved_dataset)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


st.markdown(
    """
    <div class="hero">
        <h1>Smart Farming Predictor</h1>
        <p>AI-powered crop recommendation from soil and environmental conditions</p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.sidebar.header("Input Parameters")
nitrogen = st.sidebar.slider("Nitrogen (N)", 0, 140, 80)
phosphorus = st.sidebar.slider("Phosphorus (P)", 5, 145, 40)
potassium = st.sidebar.slider("Potassium (K)", 5, 205, 40)
temperature = st.sidebar.slider("Temperature (C)", 5, 45, 25)
humidity = st.sidebar.slider("Humidity (%)", 10, 100, 70)
ph = st.sidebar.slider("Soil pH", 3.5, 9.5, 6.5, 0.1)
rainfall = st.sidebar.slider("Rainfall (mm)", 20, 300, 100)

top_k = st.sidebar.slider("Top-K results", 1, 10, 5)
run_ga_flag = st.sidebar.checkbox("Run Genetic Algorithm", value=True)
ga_generations = st.sidebar.slider(
    "GA generations",
    50,
    200,
    200,
    50,
    disabled=not run_ga_flag,
)
predict = st.sidebar.button("Predict Crop", type="primary")


if not predict:
    accuracy = get_accuracy() * 100
    columns = st.columns(4)
    metrics = [
        ("Crop classes", "22"),
        ("Training records", "2,200"),
        ("Random Forest accuracy", f"{accuracy:.2f}%"),
        ("AI techniques", "3"),
    ]
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, value)

    st.markdown("---")
    st.subheader("Available methods")
    method_columns = st.columns(3)
    method_text = [
        ("A* Search", "Ranks crop candidates using normalized Euclidean distance."),
        ("Genetic Algorithm", "Evolves candidate farming conditions over 200 generations."),
        ("Random Forest", "Provides a supervised-learning baseline prediction."),
    ]
    for column, (title, description) in zip(method_columns, method_text):
        with column:
            st.markdown(
                f'<div class="card"><strong>{title}</strong><br>{description}</div>',
                unsafe_allow_html=True,
            )

    st.info("Set the seven input parameters in the sidebar, then select Predict Crop.")

else:
    input_features = {
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }

    start = time.time()
    astar_results = astar_recommend(input_features, top_k=top_k)
    astar_time = time.time() - start

    start = time.time()
    rf_results = rf_predict(input_features, top_k=top_k)
    rf_time = time.time() - start

    ga_result = None
    ga_time = 0.0
    if run_ga_flag:
        target_crop = astar_results[0]["crop"]
        start = time.time()
        ga_result = run_ga(
            target_crop,
            generations=ga_generations,
            crossover_rate=0.8,
            seed=42,
        )
        ga_time = time.time() - start

    top_crop = astar_results[0]["crop"]

    st.success(
        f"A* recommended **{top_crop.title()}**; "
        f"Random Forest predicted **{rf_results[0]['crop'].title()}**."
    )

    tabs = st.tabs(
        ["A* and Random Forest", "Evaluation", "Genetic Algorithm"]
    )

    with tabs[0]:
        astar_column, rf_column = st.columns(2)

        with astar_column:
            st.subheader("A* Search")
            astar_table = pd.DataFrame(
                [
                    {
                        "Rank": result["rank"],
                        "Crop": result["crop"].title(),
                        "Match": f"{result['score'] * 100:.2f}%",
                        "Heuristic h(n)": result["heuristic"],
                        "Nodes expanded": result["nodes_expanded"],
                    }
                    for result in astar_results
                ]
            )
            st.dataframe(astar_table, hide_index=True, use_container_width=True)
            st.caption(f"Runtime: {astar_time * 1000:.2f} ms")

        with rf_column:
            st.subheader("Random Forest")
            rf_table = pd.DataFrame(
                [
                    {
                        "Rank": result["rank"],
                        "Crop": result["crop"].title(),
                        "Probability": f"{result['probability']:.2f}%",
                    }
                    for result in rf_results
                ]
            )
            st.dataframe(rf_table, hide_index=True, use_container_width=True)
            st.caption(
                f"Runtime: {rf_time * 1000:.2f} ms | "
                f"Accuracy: {get_accuracy() * 100:.2f}%"
            )

    with tabs[1]:
        st.subheader("Held-Out Evaluation")
        evaluation = get_evaluation()

        accuracy_table = pd.DataFrame(
            [
                {
                    "Approach": "A* Search",
                    "Accuracy": f"{evaluation['astar_accuracy'] * 100:.2f}%",
                    "Additional metric": (
                        f"{evaluation['astar_mean_nodes_expanded']:.1f} "
                        "nodes expanded"
                    ),
                },
                {
                    "Approach": "Random Forest",
                    "Accuracy": f"{evaluation['accuracy'] * 100:.2f}%",
                    "Additional metric": "440 test records",
                },
            ]
        )
        st.dataframe(
            accuracy_table,
            hide_index=True,
            use_container_width=True,
        )

        runtime_columns = st.columns(2)
        with runtime_columns[0]:
            fig_runtime = px.bar(
                pd.DataFrame(
                    [
                        {"Algorithm": "A* Search", "Time (ms)": astar_time * 1000},
                        {
                            "Algorithm": "Random Forest",
                            "Time (ms)": rf_time * 1000,
                        },
                        *(
                            [{"Algorithm": "Genetic Algorithm", "Time (ms)": ga_time * 1000}]
                            if ga_result
                            else []
                        ),
                    ]
                ),
                x="Algorithm",
                y="Time (ms)",
                title="Runtime Comparison",
            )
            st.plotly_chart(fig_runtime, use_container_width=True)

        with runtime_columns[1]:
            astar_crops = [result["crop"] for result in astar_results]
            rf_crops = [result["crop"] for result in rf_results]
            agreement = set(astar_crops).intersection(rf_crops)
            fig_agreement = go.Figure(
                go.Bar(
                    x=["A* in RF top-K", "RF in A* top-K", "Common crops"],
                    y=[
                        len(set(astar_crops).intersection(rf_crops)),
                        len(set(rf_crops).intersection(astar_crops)),
                        len(agreement),
                    ],
                )
            )
            fig_agreement.update_layout(
                title="Top-K Crop Agreement",
                yaxis_title="Crop count",
            )
            st.plotly_chart(fig_agreement, use_container_width=True)

        cm = evaluation["confusion_matrix"]
        if cm is not None:
            fig_cm = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=evaluation["classes"],
                    y=evaluation["classes"],
                    colorscale="Greens",
                    text=cm,
                    texttemplate="%{text}",
                )
            )
            fig_cm.update_layout(
                title="Random Forest Confusion Matrix",
                xaxis_title="Predicted crop",
                yaxis_title="Actual crop",
                height=560,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    with tabs[2]:
        if not ga_result:
            st.info("Enable Run Genetic Algorithm and predict again.")
        else:
            st.subheader(f"Genetic Algorithm Conditions for {top_crop.title()}")
            comparison = pd.DataFrame(
                [
                    {
                        "Feature": feature,
                        "GA optimal": ga_result["best_chromosome"][feature],
                        "Ideal": ga_result["ideal_conditions"][feature],
                    }
                    for feature in [
                        "N",
                        "P",
                        "K",
                        "temperature",
                        "humidity",
                        "ph",
                        "rainfall",
                    ]
                ]
            )
            st.dataframe(comparison, hide_index=True, use_container_width=True)
            st.caption(
                f"Best fitness: {ga_result['best_fitness']:.4f} | "
                f"Runtime: {ga_time:.2f} s"
            )

            fitness_history = pd.DataFrame(
                {
                    "Generation": range(1, len(ga_result["fitness_history"]) + 1),
                    "Best fitness": ga_result["fitness_history"],
                }
            )
            fig_fitness = px.line(
                fitness_history,
                x="Generation",
                y="Best fitness",
                title="GA Best Fitness by Generation",
            )
            st.plotly_chart(fig_fitness, use_container_width=True)

            with st.expander("Five fixed-seed GA runs"):
                experiment = run_ga_experiment(
                    top_crop,
                    generations=ga_generations,
                )
                stability = pd.DataFrame(
                    [
                        {"Seed": seed, "Best fitness": run["best_fitness"]}
                        for seed, run in zip(
                            experiment["seeds"],
                            experiment["runs"],
                        )
                    ]
                )
                st.dataframe(stability, hide_index=True, use_container_width=True)
                st.caption(
                    f"Mean fitness: {experiment['mean_fitness']:.4f} | "
                    f"Standard deviation: {experiment['std_fitness']:.4f}"
                )

    with st.expander("Input parameters"):
        input_table = pd.DataFrame(
            [
                {"Feature": "Nitrogen", "Value": nitrogen},
                {"Feature": "Phosphorus", "Value": phosphorus},
                {"Feature": "Potassium", "Value": potassium},
                {"Feature": "Temperature", "Value": temperature},
                {"Feature": "Humidity", "Value": humidity},
                {"Feature": "Soil pH", "Value": ph},
                {"Feature": "Rainfall", "Value": rainfall},
            ]
        )
        st.dataframe(input_table, hide_index=True, use_container_width=True)
