"""
AI Assignment Feedback Assistant
A teacher enters a rubric and the student's work. The AI suggests feedback and a
score for each criterion. The teacher reviews and edits everything before it is
used. Nothing is final until the teacher says so.

Design notes:
  - Rubric-based prompting: the AI only judges the criteria the teacher entered.
  - Structured responses: the AI answers in JSON, which the app validates.
  - Evidence check: every quote the AI cites is checked against the student's work.
  - Totals are added up by the code, never by the AI.
  - The student's name is never sent to the AI.
  - Document handling: PDF, DOCX and text files can be uploaded.

Uses the Groq REST API directly through `requests`, so the `groq` package is
not needed.

Run locally:  streamlit run app.py
"""

import io
import json
import os
import re
from datetime import datetime

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AI Assignment Feedback", page_icon="🧑‍🏫", layout="wide")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
TONES = ["Encouraging", "Balanced", "Direct"]
LEVELS = ["Primary school", "Middle school", "High school", "University"]
MAX_WORK_CHARS = 15000
MAX_RUBRIC_CHARS = 4000
MIN_WORK_CHARS = 50

SAMPLE_RUBRIC = """Thesis and Argument | 10 | Clear, arguable thesis that is supported consistently throughout the essay
Evidence and Examples | 10 | Specific facts, examples or sources that support each main point
Organization and Flow | 10 | Logical structure with an introduction, body paragraphs and conclusion, and smooth transitions
Grammar and Style | 10 | Correct grammar, spelling and punctuation, and clear, appropriate wording"""

SAMPLE_WORK = """Should Schools Start Later?

Many schools start very early in the morning, and I think this is a problem for students. In my opinion schools should start at 9 am or later because teenagers need more sleep.

First of all, teenagers do not get enough sleep. Many students go to bed late because of homework and phones, and then they have to wake up at 6 am. When they are tired in class they can not focus, and their grades suffer. Studies show that sleepy students make more mistakes.

Another reason is health. Students who sleep less are more likely to feel stressed and get sick. A later start time would help them feel better and be more happy at school.

Some people say that a later start means school finishes later and sports practice will be harder. This is true but I think the schools can fix it by changing the schedule. Also, parents who work early might have problem with a later start.

In conclusion, schools should start later. It is better for students sleep and health. Thats why I believe the change is a good idea."""

# HELPERS START
def parse_rubric(text):
    """Parse lines like 'Criterion | points | description' into a list of criteria."""
    rubric = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[\-\*•]+\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s+", "", line)
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        name = parts[0]
        if not name:
            continue
        max_points = 10.0
        description = ""
        if len(parts) >= 2:
            number = re.search(r"\d+(?:\.\d+)?", parts[1])
            if number:
                max_points = float(number.group(0))
                description = " | ".join(parts[2:]).strip()
            else:
                description = " | ".join(parts[1:]).strip()
        if max_points <= 0:
            max_points = 10.0
        rubric.append(
            {"id": len(rubric) + 1, "name": name, "max": max_points, "description": description}
        )
    return rubric


def fmt(number):
    """Show 10.0 as 10 and 7.5 as 7.5."""
    return str(int(number)) if float(number).is_integer() else str(number)


def normalize_text(s):
    s = str(s).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \"'.,;:-")


def quote_found(quote, work):
    """True if the quote (ignoring case, spacing and '...') really appears in the work."""
    quote = str(quote or "").strip()
    if not quote:
        return None
    haystack = normalize_text(work)
    pieces = [normalize_text(p) for p in re.split(r"\.\.\.|…", quote)]
    pieces = [p for p in pieces if len(p) >= 4]
    if not pieces:
        return False
    return all(p in haystack for p in pieces)


def build_messages(rubric, work, tone, level, title):
    system_msg = (
        "You are an experienced, fair teacher's assistant. You draft feedback on student "
        "work strictly according to the teacher's rubric. Your output is only a draft that "
        "the teacher will review. You never invent things the student did not write. "
        "Reply with valid JSON only."
    )
    rubric_lines = []
    for c in rubric:
        line = str(c["id"]) + ". " + c["name"] + " (maximum " + fmt(c["max"]) + " points)"
        if c["description"]:
            line += ": " + c["description"]
        rubric_lines.append(line)

    user_msg = (
        "Assess the STUDENT WORK against the RUBRIC.\n\n"
        + ("Assignment: " + title + "\n" if title else "")
        + "Student level: " + level + "\n"
        "Feedback tone: " + tone + "\n\n"
        "RUBRIC:\n" + "\n".join(rubric_lines) + "\n\n"
        "Return a JSON object with exactly this shape:\n"
        "{\n"
        '  "criteria": [\n'
        "    {\n"
        '      "id": the rubric number,\n'
        '      "criterion": "the criterion name",\n'
        '      "suggested_score": a number from 0 to the maximum for that criterion,\n'
        '      "evidence": "ONE short quote copied exactly from the student work that supports the score",\n'
        '      "strengths": "what the student did well for this criterion",\n'
        '      "improvements": "what would improve this criterion",\n'
        '      "feedback": "2-4 sentences written directly to the student, in the requested tone"\n'
        "    }\n"
        "  ],\n"
        '  "overall_feedback": "3-5 sentences written to the student",\n'
        '  "next_steps": ["2-4 specific, practical actions for the student"],\n'
        '  "flags": ["anything the teacher should double-check, for example work that seems off-topic, too short, or possibly not the student\'s own"]\n'
        "}\n\n"
        "Rules:\n"
        "- Include every rubric criterion exactly once, in the same order, using its number as id.\n"
        "- Base each score only on that criterion's description and on what the student actually wrote.\n"
        "- The evidence quote must be copied word for word from the student work. "
        "If there is nothing relevant to quote, use an empty string.\n"
        "- Be fair and consistent: do not give top marks unless the criterion is fully met.\n"
        "- Mention specific things from the work. Avoid generic praise.\n"
        "- Do not comment on anything the rubric does not cover, except in flags.\n"
        "- Write feedback appropriate for the student level.\n\n"
        'STUDENT WORK:\n"""\n' + work + '\n"""'
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def extract_json(text):
    """Parse JSON even if the model wrapped it in code fences or extra text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("The AI did not return valid JSON.")


def to_score(value, maximum):
    """Make a safe score between 0 and the maximum, rounded to the nearest 0.5."""
    try:
        score = float(value)
    except Exception:
        return 0.0
    score = max(0.0, min(float(maximum), score))
    return round(score * 2) / 2


def clean_str_list(items):
    out = []
    for x in items or []:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def validate(data, rubric, work):
    """Line the AI's answer up with the teacher's rubric, whatever the AI returned."""
    if not isinstance(data, dict):
        raise ValueError("The AI reply was not in the expected format.")
    by_id, by_name = {}, {}
    for it in data.get("criteria") or []:
        if not isinstance(it, dict):
            continue
        try:
            by_id[int(it.get("id"))] = it
        except Exception:
            pass
        by_name[str(it.get("criterion", "")).strip().lower()] = it

    items, matched = [], 0
    for c in rubric:
        it = by_id.get(c["id"]) or by_name.get(c["name"].lower())
        if it:
            matched += 1
        it = it or {}
        evidence = str(it.get("evidence", "") or "").strip()
        items.append(
            {
                "id": c["id"],
                "name": c["name"],
                "max": c["max"],
                "score": to_score(it.get("suggested_score"), c["max"]),
                "evidence": evidence,
                "quote_ok": quote_found(evidence, work),
                "strengths": str(it.get("strengths", "") or "").strip(),
                "improvements": str(it.get("improvements", "") or "").strip(),
                "feedback": str(it.get("feedback", "") or "").strip(),
                "assessed": bool(it),
            }
        )
    if matched == 0:
        raise ValueError("The AI reply was not in the expected format.")
    return {
        "items": items,
        "overall": str(data.get("overall_feedback", "") or "").strip(),
        "next_steps": clean_str_list(data.get("next_steps")),
        "flags": clean_str_list(data.get("flags")),
    }


def totals(scores, rubric):
    """Add up the scores in code. Returns (total, maximum, percent)."""
    total = sum(float(s) for s in scores)
    maximum = sum(c["max"] for c in rubric)
    percent = round(total / maximum * 100) if maximum else 0
    return total, maximum, percent


def build_report(title, student, rubric, scores, feedbacks, overall, next_steps, teacher_note):
    stamp = datetime.now().strftime("%Y-%m-%d")
    total, maximum, percent = totals(scores, rubric)
    lines = ["# Assignment Feedback" + (": " + title if title else ""), ""]
    if student:
        lines.append("**Student:** " + student + "  ")
    lines.append("**Date:** " + stamp + "  ")
    lines.append("**Total:** " + fmt(total) + " / " + fmt(maximum) + " (" + str(percent) + "%)")
    lines += ["", "## Criteria"]
    for c, score, fb in zip(rubric, scores, feedbacks):
        lines += ["", "### " + c["name"] + " - " + fmt(score) + " / " + fmt(c["max"]), fb or "-"]
    lines += ["", "## Overall feedback", overall or "-"]
    if next_steps:
        lines += ["", "## Next steps"] + ["- " + s for s in next_steps]
    if teacher_note.strip():
        lines += ["", "## Teacher's note", teacher_note.strip()]
    lines.append("")
    return "\n".join(lines)


def friendly_error(exc):
    msg = str(exc)
    low = msg.lower()
    if "401" in msg or "invalid api key" in low or "invalid_api_key" in low:
        return "Your Groq API key looks invalid. Please check it and try again."
    if "429" in msg or "rate limit" in low:
        return "Groq rate limit reached. Please wait a minute and try again."
    if "decommissioned" in low or "does not exist" in low or "404" in msg:
        return (
            "That model is not available on your account. Choose another model in the "
            "sidebar. Details: " + msg[:300]
        )
    if "connection" in low or "timeout" in low or "timed out" in low:
        return "Could not reach the Groq API. Check your internet connection and retry."
    if "json" in low or "expected format" in low:
        return "The AI reply could not be read. Please click Generate again."
    return "Something went wrong: " + msg
# HELPERS END


def get_api_key(sidebar_value):
    """Priority: sidebar input -> st.secrets -> environment variable."""
    if sidebar_value and sidebar_value.strip():
        return sidebar_value.strip()
    try:
        secret = st.secrets["GROQ_API_KEY"]
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_models(api_key):
    """Ask Groq which chat models this API key can use right now."""
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": "Bearer " + api_key},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        skip = ("whisper", "guard", "tts", "playai", "orpheus", "safeguard", "embed")
        ids = [
            m["id"]
            for m in r.json().get("data", [])
            if m.get("active", True) and not any(s in m["id"].lower() for s in skip)
        ]
        return sorted(ids)
    except Exception:
        return []


def call_groq(api_key, model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=120)

    # Some models do not support JSON mode: retry once without it.
    if resp.status_code == 400 and "response_format" in resp.text:
        payload.pop("response_format")
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError("Groq API error " + str(resp.status_code) + ": " + resp.text[:300])
    choices = resp.json().get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def read_document(name, data):
    """Return the text of an uploaded PDF, DOCX, TXT or MD file."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF support needs the 'pypdf' package. Add it to requirements.txt.")
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
        if not text:
            raise ValueError("No readable text found in the PDF (it may be a scanned image).")
        return text
    if lower.endswith(".docx"):
        try:
            from docx import Document
        except ImportError:
            raise ValueError("Word support needs the 'python-docx' package. Add it to requirements.txt.")
        document = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in document.paragraphs).strip()
        if not text:
            raise ValueError("No readable text found in the Word file.")
        return text
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc).strip()
        except Exception:
            continue
    raise ValueError("Could not read the file.")


# --------------------------------------------------------------------------- #
# Session state and callbacks
# --------------------------------------------------------------------------- #
for key, value in {
    "rubric_text": "",
    "work_text": "",
    "title": "",
    "file_error": "",
    "analysis": None,
    "run": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def load_example():
    st.session_state["rubric_text"] = SAMPLE_RUBRIC
    st.session_state["work_text"] = SAMPLE_WORK
    st.session_state["title"] = "Persuasive essay: Should schools start later?"
    st.session_state["file_error"] = ""


def load_work_file():
    f = st.session_state.get("work_file")
    if f is None:
        return
    try:
        st.session_state["work_text"] = read_document(f.name, f.getvalue())
        st.session_state["file_error"] = ""
    except Exception as exc:
        st.session_state["file_error"] = str(exc)


def load_rubric_file():
    f = st.session_state.get("rubric_file")
    if f is None:
        return
    try:
        st.session_state["rubric_text"] = read_document(f.name, f.getvalue())
        st.session_state["file_error"] = ""
    except Exception as exc:
        st.session_state["file_error"] = str(exc)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    key_input = st.text_input(
        "Groq API key (optional)",
        type="password",
        help="Leave empty to use st.secrets or the GROQ_API_KEY environment variable.",
    )
    api_key = get_api_key(key_input)
    if api_key:
        st.success("API key found ✅")
    else:
        st.warning("No API key found yet")

    live_models = fetch_models(api_key) if api_key else []
    model_options = live_models or MODELS
    default_index = 0
    for preferred in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
        if preferred in model_options:
            default_index = model_options.index(preferred)
            break
    model = st.selectbox("Groq model", model_options, index=default_index)

    st.subheader("Feedback style")
    tone = st.selectbox("Tone", TONES)
    level = st.selectbox("Student level", LEVELS, index=2)

    st.divider()
    st.button("📋 Load an example", on_click=load_example)
    st.caption("Fills in a sample rubric and a sample student essay.")
    st.info(
        "Privacy: the rubric and the student's work are sent to Groq. "
        "Remove names and personal details first. The student's name box below "
        "is never sent."
    )

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("🧑‍🏫 AI Assignment Feedback Assistant")
st.write(
    "Enter your rubric and the student's work. The AI drafts feedback and a score for "
    "each criterion. You review, edit and decide. Nothing is final until you say so."
)

t1, t2 = st.columns(2)
with t1:
    title = st.text_input("Assignment title (optional)", key="title")
with t2:
    student_name = st.text_input("Student name (optional, never sent to the AI)", key="student_name")

left, right = st.columns(2)
with left:
    st.subheader("1. Rubric")
    st.caption("One criterion per line: Name | maximum points | description")
    st.file_uploader(
        "Upload a rubric (optional)",
        type=["txt", "md", "pdf", "docx"],
        key="rubric_file",
        on_change=load_rubric_file,
    )
    st.text_area(
        "Rubric text",
        height=230,
        key="rubric_text",
        placeholder="Thesis and Argument | 10 | Clear, arguable thesis supported throughout",
    )
    rubric = parse_rubric(st.session_state["rubric_text"])
    if rubric:
        total_points = sum(c["max"] for c in rubric)
        with st.expander("Rubric as the app reads it: " + str(len(rubric)) + " criteria, " + fmt(total_points) + " points"):
            st.table(
                [
                    {"#": c["id"], "Criterion": c["name"], "Max": fmt(c["max"]), "Description": c["description"]}
                    for c in rubric
                ]
            )
with right:
    st.subheader("2. Student work")
    st.file_uploader(
        "Upload the work (PDF, Word or text)",
        type=["pdf", "docx", "txt", "md"],
        key="work_file",
        on_change=load_work_file,
    )
    if st.session_state["file_error"]:
        st.error(st.session_state["file_error"])
    st.text_area(
        "Student work text",
        height=300,
        key="work_text",
        placeholder="Paste the student's work here...",
    )

if st.button("✨ Generate feedback suggestions", type="primary"):
    rubric_text = st.session_state["rubric_text"].strip()
    work = st.session_state["work_text"].strip()

    if not api_key:
        st.error(
            "No Groq API key found. Enter it in the sidebar, or add GROQ_API_KEY "
            "to your Streamlit secrets."
        )
    elif not rubric:
        st.warning("Please enter a rubric first, one criterion per line.")
    elif not work:
        st.warning("Please add the student's work.")
    elif len(work) < MIN_WORK_CHARS:
        st.warning("The student's work is very short. Please check it and try again.")
    elif len(rubric_text) > MAX_RUBRIC_CHARS:
        st.warning("The rubric is too long. Please keep it under " + str(MAX_RUBRIC_CHARS) + " characters.")
    else:
        if len(work) > MAX_WORK_CHARS:
            st.warning("The work was longer than " + str(MAX_WORK_CHARS) + " characters, so only the first part was assessed.")
            work = work[:MAX_WORK_CHARS]
        try:
            with st.spinner("Reading the work against your rubric..."):
                raw = call_groq(api_key, model, build_messages(rubric, work, tone, level, title.strip()))
                result = validate(extract_json(raw), rubric, work)
            st.session_state["run"] += 1
            st.session_state["analysis"] = {
                "result": result,
                "rubric": rubric,
                "work": work,
                "title": title.strip(),
                "student": student_name.strip(),
                "run": st.session_state["run"],
            }
        except Exception as exc:
            st.error(friendly_error(exc))

# --------------------------------------------------------------------------- #
# Review and edit
# --------------------------------------------------------------------------- #
analysis = st.session_state["analysis"]
if analysis is not None:
    result = analysis["result"]
    rub = analysis["rubric"]
    run = str(analysis["run"])

    st.divider()
    st.header("3. Review and edit")
    st.warning(
        "These are AI suggestions. Check every score and comment, change anything you "
        "disagree with, and only then use them."
    )
    for flag in result["flags"]:
        st.info("🚩 Check this: " + flag)

    summary_box = st.container()

    scores, feedbacks = [], []
    for item in result["items"]:
        i = str(item["id"])
        with st.expander(item["name"] + "  (max " + fmt(item["max"]) + ")", expanded=True):
            c1, c2 = st.columns([1, 3])
            with c1:
                score = st.number_input(
                    "Score",
                    min_value=0.0,
                    max_value=float(item["max"]),
                    value=float(item["score"]),
                    step=0.5,
                    key="score_" + run + "_" + i,
                )
            with c2:
                fb = st.text_area(
                    "Feedback to the student",
                    value=item["feedback"],
                    height=110,
                    key="fb_" + run + "_" + i,
                )
            scores.append(score)
            feedbacks.append(fb)

            if not item["assessed"]:
                st.caption("⚠️ The AI did not assess this criterion. Please score it yourself.")
            if item["evidence"]:
                badge = "✅ quote found in the work" if item["quote_ok"] else "⚠️ quote NOT found in the work, check it"
                st.caption("Evidence: “" + item["evidence"] + "”  (" + badge + ")")
            if item["strengths"] or item["improvements"]:
                with st.expander("AI notes for you (not for the student)"):
                    if item["strengths"]:
                        st.markdown("**Strengths:** " + item["strengths"])
                    if item["improvements"]:
                        st.markdown("**To improve:** " + item["improvements"])

    total, maximum, percent = totals(scores, rub)
    with summary_box:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total", fmt(total) + " / " + fmt(maximum))
        m2.metric("Percentage", str(percent) + "%")
        m3.metric("Criteria", len(rub))
        st.progress(min(percent, 100) / 100)

    st.subheader("Overall feedback")
    overall = st.text_area("Overall feedback to the student", value=result["overall"], height=130, key="overall_" + run)
    steps_text = st.text_area(
        "Next steps (one per line)",
        value="\n".join(result["next_steps"]),
        height=100,
        key="steps_" + run,
    )
    note = st.text_area("Your own note (optional)", height=80, key="note_" + run)
    next_steps = [s.strip(" -•") for s in steps_text.splitlines() if s.strip(" -•")]

    st.download_button(
        "⬇️ Download feedback (.md)",
        data=build_report(analysis["title"], analysis["student"], rub, scores, feedbacks, overall, next_steps, note),
        file_name="assignment_feedback.md",
        mime="text/markdown",
    )
