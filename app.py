import streamlit as st
from graph import graph


st.set_page_config(
    page_title="Specialty Pharmacy AI Agent",
    page_icon="💊",
    layout="centered"
)

st.title("Specialty Pharmacy AI Agent")

st.write(
    "Ask questions about specialty pharmacy cases, case history, "
    "and prior authorization policy."
)

question = st.text_input(
    "Ask a question",
    placeholder="Example: What is happening with case C1001?"
)

if st.button("Ask Agent"):

    if question.strip():

        with st.spinner("Analyzing case..."):

            result = graph.invoke(
                {
                    "question": question,
                    "answer": ""
                }
            )

        st.markdown("### Agent Response")
        st.markdown(result["answer"])

    else:
        st.warning("Please enter a question.")