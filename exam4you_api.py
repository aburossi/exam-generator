import streamlit as st
import time
import json
from PyPDF2 import PdfReader
from fpdf import FPDF
from docx import Document
from openai import OpenAI

st.set_page_config(page_title="Exam Creator", page_icon="📝", layout="wide")

__version__ = "1.6.0"

# --------------------------- Helper Functions ---------------------------

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

def generate_mc_questions(client, content_text, model):
    system_prompt = (
        "You are an educator tasked with creating a high school-level multiple-choice exam. "
        "Use the given content to generate single-choice questions. Each question must have one correct answer. "
        "Generate up to 20 questions as valid JSON with this structure: "
        "[{'question': '...', 'choices': ['...'], 'correct_answer': '...', 'explanation': '...'}, ...]."
    )
    user_prompt = f"Content:\n\n{content_text}\n\nGenerate exam questions."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=16000
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"Error generating questions: {e}"

def parse_generated_questions(response):
    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start == -1 or json_end == 0:
            return None, f"No JSON data found in the response:\n{response[:500]}..."
        json_str = response[json_start:json_end]
        questions = json.loads(json_str)
        return questions, None
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}\nResponse snippet:\n{response[:500]}..."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def generate_pdf(questions, include_answers=True):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Generated Exam", ln=True, align="C")

    for i, q in enumerate(questions):
        pdf.cell(0, 10, f"Q{i+1}: {q['question']}", ln=True)
        for choice in q["choices"]:
            pdf.cell(0, 10, f" - {choice}", ln=True)
        if include_answers:
            pdf.cell(0, 10, f"Correct Answer: {q['correct_answer']}", ln=True)
            pdf.cell(0, 10, f"Explanation: {q['explanation']}", ln=True)
        pdf.ln()

    return pdf.output(dest="S").encode("latin1")

def generate_docx(questions, include_answers=True):
    doc = Document()
    doc.add_heading("Generated Exam", level=1)

    for i, q in enumerate(questions):
        doc.add_heading(f"Q{i+1}: {q['question']}", level=2)
        for choice in q["choices"]:
            doc.add_paragraph(choice, style="List Bullet")
        if include_answers:
            doc.add_paragraph(f"Correct Answer: {q['correct_answer']}")
            doc.add_paragraph(f"Explanation: {q['explanation']}")

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


# --------------------------- Main Application ---------------------------

def main():
    st.title("📝 Exam Creator")
    st.markdown(f"**Version:** {__version__}")

    if "client" not in st.session_state:
        st.session_state.client = None
    if "generated_questions" not in st.session_state:
        st.session_state.generated_questions = []

    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)

    if api_key:
        try:
            st.session_state.client = OpenAI(api_key=api_key)
            st.sidebar.success("API Key configured successfully!")
        except Exception as e:
            st.sidebar.error(f"Error initializing OpenAI client: {e}")

    mode = st.sidebar.radio("Choose Mode", ["Upload & Generate", "Take Quiz", "Download Exam"])

    if mode == "Upload & Generate":
        st.subheader("Upload File to Generate Exam")
        uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                content = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                content = extract_text_from_docx(uploaded_file)
            else:
                st.error("Unsupported file type.")
                return

            if content:
                st.text_area("Extracted Content", content[:500] + "...", height=200)
                if st.button("Generate Questions"):
                    client = st.session_state.client
                    response, error = generate_mc_questions(client, content, model)
                    if error:
                        st.error(error)
                    else:
                        questions, parse_error = parse_generated_questions(response)
                        if parse_error:
                            st.error(parse_error)
                        else:
                            st.session_state.generated_questions = questions[:20]
                            st.success(f"Generated {len(questions[:20])} questions!")

        elif mode == "Take Quiz":
            questions = st.session_state.generated_questions
            if questions:
                if "quiz_answers" not in st.session_state:
                    st.session_state.quiz_answers = [None] * len(questions)
                    st.session_state.quiz_feedback = [None] * len(questions)
                    st.session_state.correct_count = 0
        
                for i, q in enumerate(questions):
                    st.write(f"### Q{i+1}: {q['question']}")
                    # Validate choices
                    if not isinstance(q.get("choices"), list) or not q["choices"]:
                        st.error(f"Question {i+1} has invalid choices. Please regenerate the questions.")
                        continue
        
                    user_choice = st.radio(
                        f"Choose an answer for Question {i+1}:",
                        q["choices"],
                        index=-1,
                        key=f"user_choice_{i}"
                    )
        
                    if st.session_state.quiz_answers[i] is None and st.button(f"Submit Answer for Q{i+1}", key=f"submit_{i}"):
                        st.session_state.quiz_answers[i] = user_choice
                        if user_choice == q["correct_answer"]:
                            st.session_state.quiz_feedback[i] = ("Correct", q.get("explanation", ""))
                            st.session_state.correct_count += 1
                        else:
                            st.session_state.quiz_feedback[i] = (
                                "Incorrect",
                                f"The correct answer is: {q['correct_answer']}. Explanation: {q.get('explanation', '')}",
                            )
        
                    if st.session_state.quiz_answers[i] is not None:
                        feedback, explanation = st.session_state.quiz_feedback[i]
                        if feedback == "Correct":
                            st.success(f"✅ {feedback}!")
                        else:
                            st.error(f"❌ {feedback}")
                        st.write(f"**Explanation:** {explanation}")
        
                if all(answer is not None for answer in st.session_state.quiz_answers):
                    st.markdown("---")
                    st.success(f"**Your Score: {st.session_state.correct_count} / {len(questions)}**")
            else:
                st.warning("No questions generated yet.")
        
            elif mode == "Download Exam":
                questions = st.session_state.generated_questions
                if questions:
                    format_option = st.radio("Choose Format", ["PDF", "DOCX"])
                    include_answers = st.checkbox("Include Answers", value=True)
                    if st.button("Download"):
                        if format_option == "PDF":
                            file_data = generate_pdf(questions, include_answers)
                            file_name = "exam.pdf"
                        else:
                            file_data = generate_docx(questions, include_answers)
                            file_name = "exam.docx"
                        st.download_button("Download", data=file_data, file_name=file_name)
                else:
                    st.warning("No questions generated yet.")

if __name__ == "__main__":
    main()
