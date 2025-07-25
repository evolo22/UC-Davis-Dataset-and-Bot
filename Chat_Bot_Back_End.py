import dearpygui.dearpygui as dpg
import Chat_Bot_Front_End
import pandas as pd
import re



def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


df = pd.read_csv("course_data.csv")

knowledge_base = {}

for _, row in df.iterrows():
    title = row['Title'].strip()
    title_lower = title.lower()
    course_id = row['Course'].strip()
    print(course_id)
    description = row['Course Description'].strip()
    prereqs = row['Prerequisites'].strip() if pd.notna(row['Prerequisites']) else "None"
    units = str(row['Units']) if 'Units' in row else "unknown"
    terms = row['Terms Offered'].strip() if 'Terms Offered' in row and pd.notna(row['Terms Offered']) else "unspecified"

    # Generate likely questions
    

    normalized_course_id = normalize(course_id)

    question_templates = {
        f"what are the prerequisites for {normalized_course_id}": f"The prerequisites for {title} are: {prereqs}.",
        f"can you tell me what {normalized_course_id} is about": f"{title} covers the following topics: {description}",
        f"how many units is {normalized_course_id}": f"{title} is a {units}-unit course.",
        f"when is {normalized_course_id} offered": f"{title} is typically offered in the {terms}.",
        f"what is {normalized_course_id}": f"{title} is described as: {description}",
        f"can you describe {normalized_course_id}": f"Sure! {title} is described as: {description}",
        f"does {normalized_course_id} have prerequisites": f"Yes, the prerequisites for {title} are: {prereqs}.",
        f"is {normalized_course_id} hard": f"It depends on your background, but {title} has prerequisites like: {prereqs}.",
        f"who should take {normalized_course_id}": f"{title} is ideal for students interested in: {description.split('.')[0]}.",
    }

    knowledge_base.update(question_templates)

def get_answer(user_question):
    user_question = normalize(user_question)

    if user_question in knowledge_base:
        return format_answer(knowledge_base[user_question], conversational=user_question.startswith("can you") or user_question.startswith("do you know"))

    # Fallback: simple fuzzy matching
    for question in knowledge_base:
        if question in user_question:
            return format_answer(knowledge_base[question], conversational=user_question.startswith("can you") or user_question.startswith("do you know"))

    return "Sorry, I don't know the answer to that."


def format_answer(answer, conversational):
    if conversational:
        return f"Sure! Here's what I found: {answer}"
    return answer

def handle_enter(sender, app_data, user_data):
    message = dpg.get_value("user_input")
    if not message.strip():
        return

    with dpg.theme() as text_color:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (30, 30, 30, 255))

    response = get_answer(message)

    user_text = dpg.add_text(f"You: {message}", parent="Message_Box", wrap=500)
    bot_text = dpg.add_text(f"Gunrock: {response}", parent="Message_Box", wrap=500)

    dpg.bind_item_theme(user_text, text_color)
    dpg.bind_item_theme(bot_text, text_color)

    dpg.set_value("user_input", "")

    # Auto-scroll to the bottom of message box
    dpg.set_y_scroll("Message_Box", dpg.get_y_scroll_max("Message_Box"))