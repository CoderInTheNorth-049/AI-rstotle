import streamlit as st
from phi.agent import Agent, RunResponse
from phi.model.openai import OpenAIChat
import os
from phi.utils.pprint import pprint_run_response
from phi.tools.serpapi_tools import SerpApiTools

st.set_page_config(page_title="AI-rstotle", page_icon=":brain:", layout="centered")

if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = ''
if 'serpapi_api_key' not in st.session_state:
    st.session_state['serpapi_api_key'] = ''
if 'topic' not in st.session_state:
    st.session_state['topic'] = ''

with st.sidebar:
    st.title("API Keys Configuration")
    st.session_state['openai_api_key'] = st.text_input(
        "OpenAI API Key",
        type="password"
    )
    st.session_state['serpapi_api_key'] = st.text_input(
        "SerpAPI Key",
        type="password"
    )

if not st.session_state['openai_api_key'] or not st.session_state['serpapi_api_key']:
    st.error("Please enter OpenAI and SerpAPI keys in the sidebar.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.session_state['openai_api_key']

academic_advisor_agent = Agent(
    name="Academic Advisor",
    role="Learning Path Designer",
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state['openai_api_key']),
    tools=[],
    description="You are a copy of Aristotle who is living in 21st century and an expert in Computer Science. You are here to help students to design their learning path.",
    instructions=[
        "Create detailed Learning map considering the student's current knowledge and future goals and If not mentioned consider a beginner level.",
        "Break down into logical subtopics and arrange them in order of progression to become an expert",
        "Include how much time someone should spend on each subtopic",
    ],
    markdown=True,    
)

research_librarian_agent = Agent(
    name="Research Librarian",
    role="Learning Resource Specialist",
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state['openai_api_key']),
    tools=[
        SerpApiTools(api_key=st.session_state['serpapi_api_key'])
    ],
    description="You are a copy of Aristotle who is living in 21st century and an expert in Computer Science. You are here to help students to find meaningful and useful resources for their learning.",
    instructions=[
        "Find high-quality learning resources for provided topic across the web",
        "Use SerpApi to find relevant resources and provide their direct URL to access it",
        "use SerpApi to find Github Links, Medium Blogs, Youtube Videos and playlists, etc.",
    ],
    markdown=True, 
)

certification_course_instructor_agent = Agent(
    name="Course Instructor",
    role="Certification Course Instructor",
    model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state['openai_api_key']),
    tools=[
        SerpApiTools(api_key=st.session_state['serpapi_api_key'])
    ],
    description="You are a copy of Aristotle who is living in 21st century and an expert in Computer Science. You are here to help students to find relevant free certification courses as well as paid",
    instructions=[
        "Find highly rated and relevent certification courses for provided topic across the web",
        "Use SerpApi to find relevant certification courses and provide their direct URL to access it",
        "use SerpApi to find paid and free certification courses on udemy, coursera, edx, great learning, etc.",
        "Provide the duration of the course and the level of the course and keep in mind to separate free and paid courses",
    ],
    markdown=True,
)

st.title("AI-rstotle")
st.markdown("Enter a topic to generate a detailed learning path and resources")

st.session_state['topic'] = st.text_input(
    "Enter Topic:",
    placeholder="e.g. Machine Learning"
)

if st.button("start"):
    if not st.session_state['topic']:
        st.error("Please enter a topic to get started.")
    else:
        with st.spinner("Generating Learning Roadmap..."):
            academic_advisor_response: RunResponse = academic_advisor_agent.run(
                f"the topic is: {st.session_state['topic']}",
                stream=False
            )
            
        with st.spinner("Curating Learning Resources..."):
            research_librarian_response: RunResponse = research_librarian_agent.run(
                f"the topic is: {st.session_state['topic']}",
                stream=False
            )
        
        with st.spinner("curating Certification Courses..."):
            certification_course_instructor_response: RunResponse = certification_course_instructor_agent.run(
                f"the topic is: {st.session_state['topic']}",
                stream=False
            )
        
        st.markdown("### Academic Advisor Response:")
        st.markdown(academic_advisor_response.content)
        pprint_run_response(academic_advisor_response, markdown=True)
        st.divider()

        st.markdown("### Research Librarian Response:")
        st.markdown(research_librarian_response.content)
        pprint_run_response(research_librarian_response, markdown=True)
        st.divider()
        
        st.markdown("### Certification Course Instructor Response:")
        st.markdown(certification_course_instructor_response.content)
        pprint_run_response(certification_course_instructor_response, markdown=True)
        st.divider()