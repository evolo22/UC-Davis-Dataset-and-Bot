import dearpygui.dearpygui as dpg
import Chat_Bot_Front_End
import pandas as pd
import re
import difflib
from datetime import datetime
import os






data_file = "User_Questions_and_Responses.xlsx"


df = pd.DataFrame(columns=["Timestamp", "User Question", "Bot Response"])
df.to_excel(data_file, index=False)

df = pd.read_csv("course_data.csv")



data_dictionary = {}
for index, row in df.iterrows():
    list_of_info = []
    data_dictionary[row['Course']] = []
    list_of_info.append(row['Title'])
    list_of_info.append(row['Units'])
    list_of_info.append(row['Course Description'])
    list_of_info.append(row['Prerequisites'])
    list_of_info.append(row['Learning Activites'])
    list_of_info.append(row['Credit Limitations'])
    list_of_info.append(row['Grade Mode'])
    list_of_info.append(row['General Education'])
    data_dictionary[row['Course']] = list_of_info

    

    
# def get_title(course_name):
#     return data_dictionary[course_name][0]
# def get_units(course_name):
#     return data_dictionary[course_name][1]
# def get_Course_Description(course_name):
#     return data_dictionary[course_name][2]
# def get_prerequisties(course_name):
#     return data_dictionary[course_name][3]
# def get_LearningActivities(course_name):
#     return data_dictionary[course_name][4]
# def get_creditLimitations(course_name):
#     return data_dictionary[course_name][5]
# def get_gradeMode(course_name):
#     return data_dictionary[course_name][6]
# def get_GeneralED(course_name):
#     return data_dictionary[course_name][7]

def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


knowledge_base = {}


def get_requirements(str):
    
    str_list = str.split('or')
    cleaned_up_list = []
    list_of_ands = []

    for word in str_list:
        if "better" not in word and "upper division" not in word:
            cleaned_up_list.append(word.replace('\x00', '').replace('\xa0', ' '))
        if 'and' in word or ';' in word:
            list_of_ands.append(word.replace("and ", "").replace("better;", "").replace(";", "").strip())
            cleaned_up_list.append(word.replace("and ", "").replace("better;", "").replace(";", "").strip())
        # if "upper division" in word:
        #     cleaned_up_list.append("Upper Division Standing")


   
    
    cleaned_up_prereqs = ""
    for values in cleaned_up_list:
        if cleaned_up_list[0] == values:
            cleaned_up_prereqs = values.replace("C-", "")
        else:
            if values in list_of_ands:
                cleaned_up_prereqs = cleaned_up_prereqs + "\nand " + values.replace("C-", "")
            else:
                cleaned_up_prereqs = cleaned_up_prereqs + "or " + values.replace("C-", "")


    return cleaned_up_prereqs




def format_prerequisites(prereq_string):
    return get_requirements(prereq_string)


for course in data_dictionary:

    normalized_course = normalize(course)
    print(normalized_course)
    
    formatted_prereq = format_prerequisites(str(data_dictionary[course][3]))
    

    question_templates = {
        #Prerequiste based questions
        f"what are the prerequisites for {normalized_course}": f"The prerequisites for {course} are:\n{formatted_prereq}.",
        f"what classes are needed for {normalized_course}": f"The prerequisites for {course} are:\n{formatted_prereq}.",
        f"what do i need to take before {normalized_course}": f"The prerequisites for {course} are:\n{formatted_prereq}.",
        f"which courses are required for {normalized_course}": f"The prerequisites for {course} are:\n{formatted_prereq}.",
        f"what should i take before {normalized_course}": f"The prerequisites for {course} are: {formatted_prereq}.",
        f"what must i take before {normalized_course}": f"The prerequisites for {course} are: {formatted_prereq}.",
        f"what classes should i complete before {normalized_course}": f"The prerequisites for {course} are: {formatted_prereq}.",
        f"are there any requirements for {normalized_course}": f"The prerequisites for {course} are: {formatted_prereq}.",
        f"do i need any other classes before {normalized_course}": f"The prerequisites for {course} are: {formatted_prereq}.",
        f"do i need to take anything before {normalized_course}": f"The prerequisites for {course} are: {formatted_prereq}.",
        f"does {normalized_course} have prerequisites": f"Yes, the prerequisites for {course} are: {formatted_prereq}.",
        f"{normalized_course} and prerequisites": f"The prerequisites for {course} are:\n{formatted_prereq}.",
        f"is there anything i need to know before taking {normalized_course}": f"The prerequisites for {course} are: {formatted_prereq}.",
        f"does {normalized_course} have prerequisites": f"Yes, the prerequisites for {course} are: {formatted_prereq}.",

        #Course Description Questions
        f"what is {normalized_course}": f"{course} is described as: {str(data_dictionary[course][2])}",
        f"can you describe {normalized_course}": f"Sure! {course} is described as: {str(data_dictionary[course][2])}",
        f"what does {normalized_course} cover": f"{course} covers the following topics: {str(data_dictionary[course][2])}",
        f"can you tell me what {normalized_course} is about": f"{course} covers the following topics: {str(data_dictionary[course][2])}",
        f"what topics are in {normalized_course}": f"{course} covers the following topics: {str(data_dictionary[course][2])}",
        f"what do you learn in {normalized_course}": f"{course} covers the following topics: {str(data_dictionary[course][2])}",
        f"what will i study in {normalized_course}": f"{course} covers the following topics: {str(data_dictionary[course][2])}",
        f"what kind of material does {normalized_course} teach": f"{course} covers the following topics: {str(data_dictionary[course][2])}",
        f"can you tell me what {normalized_course} is about": f"{course} covers the following topics: {str(data_dictionary[course][2])}",
        f"can you describe {normalized_course}": f"Sure! {course} is described as: {str(data_dictionary[course][2])}",
        f"what is {normalized_course}": f"{course} is described as: {str(data_dictionary[course][2])}",

        
        #Units Questions
        f"how many units is {normalized_course}": f"{course} is a {str(data_dictionary[course][2])}-unit course.",
        f"how many credits is {normalized_course}": f"{course} is a {str(data_dictionary[course][2])}-unit course.",
        f"what is the unit count for {normalized_course}": f"{course} is a {str(data_dictionary[course][2])}-unit course.",
        f"how much credit do you get for {normalized_course}": f"{course} is a {str(data_dictionary[course][2])}-unit course.",
        f"when is {normalized_course} offered": f"{course} is typically offered in the {str(data_dictionary[course][2])}.",

        #Learning Activities Questions

        f"what are the learning activities in {normalized_course}": f"The learning activities for {course} are: {str(data_dictionary[course][4])}.",
        f"how many hours of lecture and lab in {normalized_course}": f"The learning activities for {course} include: {str(data_dictionary[course][4])}.",
        f"what is the schedule like for {normalized_course}": f"The learning activities for {course} include: {str(data_dictionary[course][4])}.",
        f"how is {normalized_course} structured": f"The learning activities for {course} are: {str(data_dictionary[course][4])}.",
        f"how much lab or lecture time is in {normalized_course}": f"The learning activities for {course} are: {str(data_dictionary[course][4])}.",
        f"does {normalized_course} have a lab or discussion section": f"The learning activities for {course} are: {str(data_dictionary[course][4])}.",

        

        #Credit Limitations Questions

        f"are there any credit limitations for {normalized_course}": f"The credit limitations for {course} are: {str(data_dictionary[course][5])}.",
        f"who can take {normalized_course}": f"The credit limitations for {course} are: {str(data_dictionary[course][5])}.",
        f"can i take {normalized_course} if i already took another class": f"The credit limitations for {course} are: {str(data_dictionary[course][5])}.",
        f"will i get credit for {normalized_course} if i took a similar class": f"The credit limitations for {course} are: {str(data_dictionary[course][5])}.",
        f"can i repeat {normalized_course} for credit": f"The credit limitations for {course} are: {str(data_dictionary[course][5])}.",



        #Grade Mode Questions

        f"how is {normalized_course} graded": f"{course} uses the following grade mode: {str(data_dictionary[course][6])}.",
        f"what grade mode is used in {normalized_course}": f"{course} uses the following grade mode: {str(data_dictionary[course][6])}.",
        f"what is the grade mode for {normalized_course}": f"{course} uses the following grade mode: {str(data_dictionary[course][6])}.",
        f"can you take {normalized_course} pass no pass": f"{course} uses the following grade mode: {str(data_dictionary[course][6])}.",
        f"is {normalized_course} graded or pass fail": f"{course} uses the following grade mode: {str(data_dictionary[course][6])}.",


        #General Eductation Questions

        f"what general education requirements does {normalized_course} fulfill": f"{course} satisfies the following GE requirements: {str(data_dictionary[course][7])}.",
        f"does {normalized_course} count for any GEs": f"{course} satisfies the following GE requirements: {str(data_dictionary[course][7])}.",
        f"which GE areas does {normalized_course} satisfy": f"{course} satisfies the following GE requirements: {str(data_dictionary[course][7])}.",
        f"can i take {normalized_course} for GE credit": f"{course} satisfies the following GE requirements: {str(data_dictionary[course][7])}.",
        f"what topical breadths does {normalized_course} fulfill": f"{course} satisfies the following GE requirements: {str(data_dictionary[course][7])}.",


    }

    knowledge_base.update(question_templates)

casual_responses = {
"hello": "Hi there! How can I help you today?",
"hi": "Hello! What would you like to know?",
"hi gunrock": "Hey hey! What do you need help with today?",
"hey": "Hey! Need help with a course?",
"how are you": "I'm just a bunch of code, but I'm feeling helpful!",
"how's it going": "I'm here and ready to help you with your classes!",
"good morning": "Good morning! What can I do for you?",
"good afternoon": "Good afternoon! Got any course questions?",
"good evening": "Good evening! I'm here to assist you.",
"thank you": "You're welcome!",
"thanks": "Anytime!",
"who are you": "I'm Gunrock, your UC Davis course helper bot!",
"what can you do": "I can help you find course prerequisites, descriptions, units, and more!",
    }

knowledge_base.update(casual_responses)


answer_responses = ["yes", "yeh", "ye", "sure", "no", "nah", "nope"]


previous_answer = ""
previous_question = ""
    
def processes_response(answer, question):

    if answer == 'no' or answer == 'nah' or answer == 'nope':
        return "Alrighty, I will be here when you need help"
    else:
        return "Sure, with what course and what about?"


def get_answer(user_question):

    global previous_answer, previous_question


    user_question = normalize(user_question)

    if user_question in answer_responses:
        new_answer = processes_response(previous_answer, previous_question)
        return new_answer



    close_matches = difflib.get_close_matches(user_question, knowledge_base.keys(), cutoff=0.8)

    if close_matches == []:
        close_matches = difflib.get_close_matches(user_question, knowledge_base.keys(), cutoff=0.7)


    print(close_matches)

    if close_matches:
        previous_answer = format_answer(knowledge_base[close_matches[0]])
        previous_question = user_question
        return previous_answer
    

    return "I'm not sure yet, but I'll try to learn! Is there anything else I can help you with?"


def format_answer(answer):

    casual_responses = {
"hello": "Hi there! How can I help you today?",
"hi": "Hello! What would you like to know?",
"hi gunrock": "Hey hey! What do you need help with today?",
"hey": "Hey! Need help with a course?",
"how are you": "I'm just a bunch of code, but I'm feeling helpful!",
"how's it going": "I'm here and ready to help you with your classes!",
"good morning": "Good morning! What can I do for you?",
"good afternoon": "Good afternoon! Got any course questions?",
"good evening": "Good evening! I'm here to assist you.",
"thank you": "You're welcome!",
"thanks": "Anytime!",
"who are you": "I'm Gunrock, your UC Davis course helper bot!",
"what can you do": "I can help you find course prerequisites, descriptions, units, and more!",
    }

    if answer in casual_responses.values():
        return answer
    else:
        return answer + "\nIs there anything else I can help you with?"
    

def handle_enter(sender, app_data, user_data):
    message = dpg.get_value("user_input")

    with dpg.theme() as text_color:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (30, 30, 30, 255))

    response = get_answer(message)


    log_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User Question": message,
        "Bot Response": response
    }])


    existing_log = pd.read_excel(data_file)
    updated_log = pd.concat([existing_log, log_entry], ignore_index=True)
    updated_log.to_excel(data_file, index=False)

    user_text = dpg.add_text(f"You: {message}", parent="Message_Box", wrap=500)
    bot_text = dpg.add_text(f"Gunrock: {response}", parent="Message_Box", wrap=500)

    dpg.bind_item_theme(user_text, text_color)
    dpg.bind_item_theme(bot_text, text_color)

    dpg.set_value("user_input", "")
    dpg.set_y_scroll("Message_Box", dpg.get_y_scroll_max("Message_Box"))