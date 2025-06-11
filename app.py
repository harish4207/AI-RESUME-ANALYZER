import streamlit as st
import spacy
import fitz  # PyMuPDF
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import io
from docx import Document
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font and text color */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #333333 !important; /* Force dark text color for all Streamlit elements */
    }

    .main {
        padding: 2rem;
        background-color: #f0f2f6; /* Light gray background */
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: #f0f2f6; /* Light gray background */
    }
    .header {
        text-align: center;
        padding: 2.5rem 0;
        background: linear-gradient(90deg, #2196F3, #1976D2); /* Brighter blue gradient */
        color: white !important; /* Ensure header text is white */
        border-radius: 12px;
        margin-bottom: 2.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .header h1 {
        font-size: 2.8rem;
        margin: 0;
        padding: 0;
        font-weight: 700;
    }
    .header p {
        font-size: 1.3rem;
        margin: 0.8rem 0 0 0;
        opacity: 0.9;
        font-weight: 400;
    }
    .metric-card {
        background-color: white;
        padding: 1.8rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
        border-top: 5px solid #2196F3;
    }
    .suggestion-card {
        background-color: white; /* White background for suggestions */
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin: 0.7rem 0;
        border-left: 5px solid #4CAF50; /* Green border for suggestions */
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        font-size: 1.05rem;
        line-height: 1.6;
        color: #333333 !important; /* Ensure text color is dark */
    }
    .upload-section {
        background-color: white; /* White background for input sections */
        padding: 2.2rem;
        border-radius: 12px;
        margin-bottom: 2.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        height: 100%; /* Ensure equal height */
        display: flex;
        flex-direction: column;
    }
    .upload-section h2 {
        color: #333333 !important; /* Ensure heading text is dark */
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4CAF50, #388E3C); /* Green gradient for buttons */
        color: white !important; /* Ensure button text is white */
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #388E3C, #2E7D32);
        transform: translateY(-2px);
    }
    .tab-content {
        padding: 1.5rem 0;
    }
    .highlight {
        background-color: #E8F5E9; /* Light green for highlight */
        padding: 0.2rem 0.6rem;
        border-radius: 5px;
        font-weight: 600;
        color: #2E7D32 !important; /* Ensure highlight text is dark green */
    }
    /* More specific and forceful styles for input fields */
    .stTextInput input[type="text"],
    .stTextArea textarea {
        background-color: #ffffff !important; /* White background for input boxes */
        color: #333333 !important; /* Dark text for readability */
        border-radius: 8px;
        border: 1px solid #cccccc;
        padding: 0.75rem;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.06);
    }
    /* Styles for the file uploader */
    .stFileUploader label {
        color: #333333 !important; /* Ensure label text is dark */
    }
    .stFileUploader div[data-testid="stFileUploadDropzone"] {
        background-color: #ffffff !important; /* White background for drop area */
        color: #333333 !important; /* Dark text for readability */
        border: 2px dashed #cccccc;
        border-radius: 8px;
        padding: 1rem;
    }
    .stFileUploader div[data-testid="stFileUploadDropzone"] p {
        color: #333333 !important; /* Ensure paragraph text inside dropzone is dark */
    }
    .stFileUploader button {
        background-color: #f8f8f8 !important;
        color: #333333 !important; /* Dark text for file uploader button */
        border-radius: 8px;
        border: 1px solid #cccccc;
        padding: 0.5rem;
    }
    .stFileUploader div[data-testid="stUploadedFile"] div[data-testid="stFileDropzoneFileBox"] span {
        color: #333333 !important; /* Ensure uploaded file name text is dark */
    }
    </style>
    """, unsafe_allow_html=True)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

def is_valid_resume(text):
    """Check if the uploaded file contains resume-like content."""
    # Minimum length check
    if len(text.split()) < 50:
        return False, "The file seems too short to be a resume. Please upload a complete resume."
    
    # Check for common resume sections
    required_sections = ['experience', 'education', 'skills']
    found_sections = [section for section in required_sections if section.lower() in text.lower()]
    if len(found_sections) < 2:
        return False, "The file doesn't appear to be a resume. It should contain sections like Experience, Education, and Skills."
    
    # Check for contact information
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    has_phone = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
    if not (has_email or has_phone):
        return False, "The file doesn't contain contact information (email or phone number). Please upload a complete resume."
    
    return True, "Valid resume format detected."

def extract_text_from_file(uploaded_file):
    """Extract text from various file formats."""
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text
    except Exception as e:
        raise ValueError("Error reading PDF file. Please ensure it's a valid PDF document.")

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file."""
    try:
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        raise ValueError("Error reading DOCX file. Please ensure it's a valid Word document.")

def clean_text(text):
    """Clean and preprocess text."""
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_keywords(text):
    """Extract keywords using spaCy."""
    doc = nlp(text)
    keywords = []
    
    # Extract named entities, nouns, and proper nouns
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART']:
            keywords.append(ent.text.lower())
    
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
            keywords.append(token.text.lower())
    
    return list(set(keywords))

def calculate_similarity(resume_keywords, job_keywords):
    """Calculate similarity between resume and job keywords."""
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([' '.join(resume_keywords), ' '.join(job_keywords)])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100
    except:
        return 0

def calculate_ats_score(resume_text):
    """Calculate overall ATS score based on various factors."""
    scores = {
        'format': 0,
        'keywords': 0,
        'contact': 0,
        'readability': 0
    }
    
    # Format score (check for common sections)
    sections = ['experience', 'education', 'skills', 'summary', 'objective']
    format_score = sum(1 for section in sections if section.lower() in resume_text.lower()) * 20
    scores['format'] = min(format_score, 100)
    
    # Keywords score (based on keyword density and variety)
    words = resume_text.lower().split()
    word_count = len(words)
    unique_words = len(set(words))
    keyword_score = (unique_words / word_count * 100) if word_count > 0 else 0
    scores['keywords'] = min(keyword_score, 100)
    
    # Contact information score
    contact_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # phone
        r'\b(linkedin\.com|github\.com)\b'  # social profiles
    ]
    contact_score = sum(1 for pattern in contact_patterns if re.search(pattern, resume_text)) * 33.33
    scores['contact'] = min(contact_score, 100)
    
    # Readability score
    sentences = re.split(r'[.!?]+', resume_text)
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    readability_score = 100 - (abs(avg_sentence_length - 15) * 5)  # Optimal sentence length is 15 words
    scores['readability'] = max(0, min(readability_score, 100))
    
    # Calculate overall score
    overall_score = sum(scores.values()) / len(scores)
    
    return {
        'overall': overall_score,
        'details': scores
    }

def get_industry_specific_suggestions(resume_text):
    """Generate industry-specific suggestions based on content analysis."""
    suggestions = []
    
    # Define industry-specific keywords
    industries = {
        'tech': ['software', 'programming', 'development', 'coding', 'algorithm', 'database', 'api', 'cloud'],
        'finance': ['financial', 'accounting', 'budget', 'revenue', 'investment', 'analysis', 'forecast'],
        'marketing': ['marketing', 'campaign', 'social media', 'content', 'strategy', 'brand', 'digital'],
        'healthcare': ['medical', 'healthcare', 'patient', 'clinical', 'treatment', 'health'],
        'education': ['teaching', 'education', 'curriculum', 'student', 'learning', 'instruction']
    }
    
    # Detect potential industry
    industry_scores = {industry: 0 for industry in industries}
    for industry, keywords in industries.items():
        for keyword in keywords:
            if keyword in resume_text.lower():
                industry_scores[industry] += 1
    
    # Get top industry
    top_industry = max(industry_scores.items(), key=lambda x: x[1])
    if top_industry[1] > 0:
        suggestions.append(f"Resume appears to be in the {top_industry[0]} industry. Consider adding more industry-specific keywords.")
    
    return suggestions

def get_achievement_analysis(resume_text):
    """Analyze achievements and provide specific suggestions."""
    suggestions = []
    
    # Check for achievement indicators
    achievement_indicators = [
        r'increased by \d+%',
        r'decreased by \d+%',
        r'reduced by \d+%',
        r'improved by \d+%',
        r'managed \d+',
        r'led \d+',
        r'generated \$?\d+',
        r'saved \$?\d+',
        r'completed \d+',
        r'developed \d+'
    ]
    
    found_achievements = []
    for pattern in achievement_indicators:
        matches = re.findall(pattern, resume_text.lower())
        found_achievements.extend(matches)
    
    if len(found_achievements) < 3:
        suggestions.append("Add more quantifiable achievements with specific numbers and percentages")
    
    # Check for STAR method usage
    star_components = {
        'situation': ['faced with', 'when', 'during', 'while'],
        'task': ['responsible for', 'tasked with', 'assigned to'],
        'action': ['implemented', 'developed', 'created', 'designed'],
        'result': ['resulted in', 'led to', 'achieved', 'accomplished']
    }
    
    missing_components = []
    for component, keywords in star_components.items():
        if not any(keyword in resume_text.lower() for keyword in keywords):
            missing_components.append(component)
    
    if missing_components:
        suggestions.append(f"Use the STAR method more effectively by including {', '.join(missing_components)} in your achievements")
    
    return suggestions

def get_format_suggestions(format_score, resume_text):
    """Generate specific suggestions for improving resume format."""
    suggestions = []
    sections = {
        'experience': ['experience', 'work experience', 'employment', 'professional experience'],
        'education': ['education', 'academic', 'qualification', 'degree'],
        'skills': ['skills', 'technical skills', 'competencies', 'expertise', 'proficiencies'],
        'summary': ['summary', 'profile', 'professional summary', 'career objective'],
        'contact': ['contact', 'contact information', 'contact details']
    }
    
    # Check for missing sections
    missing_sections = []
    for section, keywords in sections.items():
        if not any(keyword in resume_text.lower() for keyword in keywords):
            missing_sections.append(section)
    
    if missing_sections:
        suggestions.append(f"Add missing sections: {', '.join(missing_sections)}")
    
    # Check section order
    section_positions = {}
    for section, keywords in sections.items():
        for keyword in keywords:
            pos = resume_text.lower().find(keyword)
            if pos != -1:
                section_positions[section] = pos
                break
    
    if section_positions:
        ordered_sections = sorted(section_positions.items(), key=lambda x: x[1])
        current_order = [s[0] for s in ordered_sections]
        ideal_order = ['contact', 'summary', 'experience', 'education', 'skills']
        
        if current_order != ideal_order:
            suggestions.append(f"Reorganize sections in this order: {' → '.join(ideal_order)}")
    
    # Check for formatting
    lines = resume_text.split('\n')
    bullet_points = sum(1 for line in lines if re.match(r'^[\s]*[•\-\*]', line))
    if bullet_points < 3:
        suggestions.append("Add more bullet points to highlight achievements and responsibilities")
    
    # Check for consistent date formatting
    date_patterns = [
        r'\d{4}-\d{2}',
        r'\d{2}/\d{2}/\d{4}',
        r'\d{2}-\d{2}-\d{4}',
        r'[A-Z][a-z]+ \d{4}'
    ]
    found_dates = []
    for pattern in date_patterns:
        found_dates.extend(re.findall(pattern, resume_text))
    
    if len(set(found_dates)) > 1:
        suggestions.append("Use consistent date formatting throughout your resume")
    
    return suggestions

def get_keyword_suggestions(keyword_score, resume_text):
    """Generate suggestions for improving keyword optimization."""
    suggestions = []
    
    # Analyze keyword density and variety
    words = resume_text.lower().split()
    word_count = len(words)
    unique_words = len(set(words))
    keyword_density = (unique_words / word_count * 100) if word_count > 0 else 0
    
    # Check for action verbs
    action_verbs = {
        'achievement': ['achieved', 'accomplished', 'attained', 'completed', 'delivered'],
        'leadership': ['led', 'managed', 'directed', 'supervised', 'coordinated'],
        'improvement': ['improved', 'enhanced', 'optimized', 'increased', 'developed'],
        'technical': ['developed', 'implemented', 'designed', 'created', 'programmed'],
        'communication': ['presented', 'communicated', 'collaborated', 'negotiated', 'facilitated'],
        'analysis': ['analyzed', 'evaluated', 'assessed', 'researched', 'investigated']
    }
    
    found_verbs = {category: [] for category in action_verbs}
    for category, verbs in action_verbs.items():
        for verb in verbs:
            if verb in resume_text.lower():
                found_verbs[category].append(verb)
    
    # Provide specific suggestions based on found verbs
    missing_categories = [cat for cat, verbs in found_verbs.items() if not verbs]
    if missing_categories:
        suggestions.append(f"Add more {', '.join(missing_categories)} action verbs to strengthen your achievements")
    
    # Check for quantified achievements
    numbers = re.findall(r'\b\d+\b', resume_text)
    if len(numbers) < 3:
        suggestions.append("Add more quantified achievements (e.g., 'increased sales by 25%', 'managed team of 10')")
    
    # Check for technical skills
    technical_terms = [
        'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 'node.js',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'agile', 'scrum', 'jira'
    ]
    found_technical = [term for term in technical_terms if term in resume_text.lower()]
    if len(found_technical) < 3:
        suggestions.append("List more specific technical skills and technologies you're proficient in")
    
    # Check for soft skills
    soft_skills = [
        'communication', 'leadership', 'teamwork', 'problem-solving', 'time management',
        'adaptability', 'creativity', 'critical thinking', 'emotional intelligence'
    ]
    found_soft_skills = [skill for skill in soft_skills if skill in resume_text.lower()]
    if len(found_soft_skills) < 2:
        suggestions.append("Include more soft skills that are relevant to your target role")
    
    return suggestions

def get_contact_suggestions(contact_score, resume_text):
    """Generate suggestions for improving contact information."""
    suggestions = []
    
    # Check for email
    email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
    if not email:
        suggestions.append("Add your professional email address")
    elif email.group().lower().endswith(('@gmail.com', '@yahoo.com', '@hotmail.com')):
        suggestions.append("Consider using a professional email address instead of a personal one")
    
    # Check for phone
    phone = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', resume_text)
    if not phone:
        suggestions.append("Add your phone number")
    
    # Check for LinkedIn
    linkedin = re.search(r'linkedin\.com', resume_text)
    if not linkedin:
        suggestions.append("Include your LinkedIn profile URL")
    
    # Check for location
    location = re.search(r'\b[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*,\s*[A-Z]{2}\b', resume_text)
    if not location:
        suggestions.append("Add your city and state")
    
    # Check for portfolio/website
    website = re.search(r'(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/\S*)?', resume_text)
    if not website:
        suggestions.append("Consider adding your portfolio or personal website if applicable")
    
    # Check for social media presence
    social_media = {
        'GitHub': 'github.com',
        'Twitter': 'twitter.com',
        'Medium': 'medium.com',
        'Stack Overflow': 'stackoverflow.com'
    }
    
    found_social = [platform for platform, domain in social_media.items() 
                   if domain in resume_text.lower()]
    if not found_social:
        suggestions.append("Consider adding relevant professional social media profiles")
    
    return suggestions

def get_readability_suggestions(readability_score, resume_text):
    """Generate suggestions for improving readability."""
    suggestions = []
    
    # Analyze sentence length
    sentences = re.split(r'[.!?]+', resume_text)
    long_sentences = [s for s in sentences if len(s.split()) > 20]
    if long_sentences:
        suggestions.append(f"Break down {len(long_sentences)} long sentences into shorter, more concise ones")
    
    # Check bullet points
    lines = resume_text.split('\n')
    bullet_points = sum(1 for line in lines if re.match(r'^[\s]*[•\-\*]', line))
    if bullet_points < 5:
        suggestions.append("Use more bullet points to list achievements and responsibilities")
    
    # Check for consistent formatting
    headers = re.findall(r'[A-Z][a-z]+:', resume_text)
    if len(set(headers)) < 3:
        suggestions.append("Use consistent formatting for section headers (e.g., 'Experience:', 'Education:', 'Skills:')")
    
    # Check for proper spacing
    double_spaces = re.findall(r'\s{2,}', resume_text)
    if double_spaces:
        suggestions.append("Remove extra spaces between words and sections")
    
    # Check for proper capitalization
    words = resume_text.split()
    proper_nouns = [word for word in words if word[0].isupper() and len(word) > 1]
    if len(proper_nouns) < 5:
        suggestions.append("Ensure proper capitalization of company names, job titles, and proper nouns")
    
    # Check for consistent tense
    past_tense = re.findall(r'\b(?:ed|d)\b', resume_text.lower())
    present_tense = re.findall(r'\b(?:ing|s)\b', resume_text.lower())
    if past_tense and present_tense:
        suggestions.append("Use consistent tense throughout your resume (preferably past tense for past experiences)")
    
    return suggestions

def get_role_specific_skills():
    """Return a mapping of job roles to their required skills."""
    return {
        'web development': {
            'frontend': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'typescript', 'sass', 'bootstrap', 'tailwind'],
            'backend': ['node.js', 'express', 'python', 'django', 'flask', 'php', 'laravel', 'java', 'spring', 'ruby', 'rails'],
            'database': ['mysql', 'postgresql', 'mongodb', 'redis', 'sql', 'nosql'],
            'devops': ['git', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd', 'jenkins'],
            'tools': ['webpack', 'npm', 'yarn', 'git', 'vscode', 'postman']
        },
        'data science': {
            'programming': ['python', 'r', 'sql', 'scala', 'java'],
            'libraries': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras'],
            'tools': ['jupyter', 'tableau', 'power bi', 'excel', 'spark', 'hadoop'],
            'techniques': ['machine learning', 'deep learning', 'statistical analysis', 'data visualization', 'nlp']
        },
        'mobile development': {
            'ios': ['swift', 'objective-c', 'xcode', 'cocoa', 'ios sdk'],
            'android': ['kotlin', 'java', 'android studio', 'gradle', 'android sdk'],
            'cross-platform': ['react native', 'flutter', 'xamarin', 'ionic'],
            'tools': ['git', 'firebase', 'app store', 'play store', 'testflight']
        },
        'devops': {
            'cloud': ['aws', 'azure', 'gcp', 'digital ocean', 'heroku'],
            'containers': ['docker', 'kubernetes', 'rancher', 'openshift'],
            'ci/cd': ['jenkins', 'gitlab ci', 'github actions', 'circleci', 'travis ci'],
            'monitoring': ['prometheus', 'grafana', 'elk stack', 'new relic', 'datadog'],
            'infrastructure': ['terraform', 'ansible', 'puppet', 'chef', 'cloudformation']
        },
        'cybersecurity': {
            'security': ['network security', 'application security', 'cloud security', 'endpoint security'],
            'tools': ['wireshark', 'nmap', 'metasploit', 'burp suite', 'nessus'],
            'frameworks': ['nist', 'iso 27001', 'pci dss', 'gdpr'],
            'certifications': ['ceh', 'cissp', 'security+', 'oscp']
        },
        'ui/ux design': {
            'design tools': ['figma', 'sketch', 'adobe xd', 'illustrator', 'photoshop'],
            'prototyping': ['invision', 'marvel', 'proto.io', 'axure'],
            'principles': ['user research', 'wireframing', 'prototyping', 'usability testing'],
            'technologies': ['html', 'css', 'javascript', 'responsive design']
        },
        'project management': {
            'methodologies': ['agile', 'scrum', 'waterfall', 'kanban', 'lean'],
            'tools': ['jira', 'trello', 'asana', 'monday.com', 'ms project'],
            'certifications': ['pmp', 'prince2', 'scrum master', 'pmi-acp'],
            'skills': ['risk management', 'budgeting', 'stakeholder management', 'team leadership']
        },
        'business analysis': {
            'tools': ['jira', 'confluence', 'visio', 'power bi', 'tableau'],
            'methodologies': ['agile', 'waterfall', 'bpmn', 'uml'],
            'certifications': ['cbap', 'ccba', 'pmi-pba'],
            'skills': ['requirements gathering', 'process modeling', 'stakeholder management', 'data analysis']
        },
        'digital marketing': {
            'channels': ['seo', 'sem', 'social media', 'email marketing', 'content marketing'],
            'tools': ['google analytics', 'google ads', 'facebook ads', 'hubspot', 'mailchimp'],
            'skills': ['content creation', 'analytics', 'campaign management', 'social media management'],
            'certifications': ['google ads', 'hubspot', 'facebook blueprint']
        }
    }

def detect_job_role(job_description):
    """Detect the primary job role from the job description."""
    job_description = job_description.lower()
    role_scores = {}
    
    for role, skills in get_role_specific_skills().items():
        score = 0
        # Check for role-specific keywords
        if role in job_description:
            score += 2
        # Check for skills mentioned
        for category, skill_list in skills.items():
            for skill in skill_list:
                if skill in job_description:
                    score += 1
        role_scores[role] = score
    
    # Return the role with the highest score
    if role_scores:
        return max(role_scores.items(), key=lambda x: x[1])[0]
    return None

def get_skill_suggestions(resume_text, job_description):
    """Generate skill suggestions based on job description and resume content."""
    suggestions = []
    resume_text = resume_text.lower()
    job_description = job_description.lower()
    
    # Detect job role
    role = detect_job_role(job_description)
    if not role:
        return suggestions
    
    # Get role-specific skills
    role_skills = get_role_specific_skills().get(role, {})
    
    # Check for missing skills in each category
    for category, skills in role_skills.items():
        missing_skills = []
        for skill in skills:
            if skill not in resume_text:
                missing_skills.append(skill)
        
        if missing_skills:
            suggestions.append(f"Consider adding these {category} skills: {', '.join(missing_skills[:5])}")
    
    return suggestions

def get_ats_improvement_suggestions(ats_scores, resume_text, job_description=None):
    """Generate comprehensive ATS improvement suggestions."""
    suggestions = {
        'format': get_format_suggestions(ats_scores['details']['format'], resume_text),
        'keywords': get_keyword_suggestions(ats_scores['details']['keywords'], resume_text),
        'contact': get_contact_suggestions(ats_scores['details']['contact'], resume_text),
        'readability': get_readability_suggestions(ats_scores['details']['readability'], resume_text)
    }
    
    # Add industry-specific suggestions
    suggestions['industry'] = get_industry_specific_suggestions(resume_text)
    
    # Add achievement analysis
    suggestions['achievements'] = get_achievement_analysis(resume_text)
    
    # Add role-specific skill suggestions if job description is provided
    if job_description:
        suggestions['skills'] = get_skill_suggestions(resume_text, job_description)
    
    return suggestions

def create_gauge_chart(score, title):
    """Create a gauge chart for score visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': f"<span style='font-size:1.2em; color:#333333;'>{title}</span>"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2196F3"},
            'steps': [
                {'range': [0, 50], 'color': "#FFCDD2"},
                {'range': [50, 75], 'color': "#FFF9C4"},
                {'range': [75, 100], 'color': "#C8E6C9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), font=dict(color="#333333", family="Inter"))
    return fig

def create_keyword_chart(matching_keywords, missing_keywords):
    """Create a bar chart for keyword analysis."""
    df = pd.DataFrame({
        'Category': ['Matching Keywords', 'Missing Keywords'],
        'Count': [len(matching_keywords), len(missing_keywords)]
    })
    fig = px.bar(df, x='Category', y='Count',
                 color='Category',
                 color_discrete_sequence=['#4CAF50', '#F44336'],
                 text_auto=True)
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color="#333333", family="Inter"),
        title_text='Keyword Distribution',
        title_x=0.5,
        yaxis_title="Number of Keywords",
        xaxis_title=""
    )
    return fig

def main():
    # Header
    st.markdown("""
        <div class="header">
            <h1>AI Resume Analyzer</h1>
            <p>Upload your resume and paste a job description to get professional analysis and suggestions</p>
        </div>
    """, unsafe_allow_html=True)

    # Create two columns for the main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
            <div class="upload-section">
                <h2><span style='font-size:1.5em; margin-right: 10px;'>📄</span>Upload Resume</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'txt', 'docx'],
            help="Supported formats: PDF, TXT, DOCX"
        )

    with col2:
        st.markdown("""
            <div class="upload-section">
                <h2><span style='font-size:1.5em; margin-right: 10px;'>🎯</span>Job Description</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Job description input
        job_description = st.text_area(
            "Paste the job description here",
            height=200,
            placeholder="e.g., Senior Software Engineer with expertise in Python and AWS...",
            help="Enter the job description to analyze resume compatibility"
        )

    if uploaded_file:
        if st.button("Analyze Resume", key="analyze_button"):
            try:
                # Process resume
                resume_text = extract_text_from_file(uploaded_file)
                resume_text = clean_text(resume_text)
                
                # Validate resume content
                is_valid, message = is_valid_resume(resume_text)
                if not is_valid:
                    st.error(message)
                    return
                
                resume_keywords = extract_keywords(resume_text)
                
                # Calculate ATS score
                ats_scores = calculate_ats_score(resume_text)
                
                # Create tabs for different sections
                tab1, tab2, tab3 = st.tabs(["📊 Score Analysis", "📝 Improvement Suggestions", "🎯 Job Match"])
                
                with tab1:
                    st.markdown("### Overall ATS Score")
                    st.plotly_chart(create_gauge_chart(ats_scores['overall'], "Overall Score"), use_container_width=True)
                    
                    # Detailed scores in a grid
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Format Score", f"{ats_scores['details']['format']:.1f}%")
                    with col2:
                        st.metric("Keywords Score", f"{ats_scores['details']['keywords']:.1f}%")
                    with col3:
                        st.metric("Contact Score", f"{ats_scores['details']['contact']:.1f}%")
                    with col4:
                        st.metric("Readability Score", f"{ats_scores['details']['readability']:.1f}%")
                
                with tab2:
                    # Get improvement suggestions
                    suggestions = get_ats_improvement_suggestions(ats_scores, resume_text, job_description)
                    
                    # Format suggestions
                    st.markdown("### Structure and Format")
                    for suggestion in suggestions['format']:
                        st.markdown(f'<div class="suggestion-card">{suggestion}</div>', unsafe_allow_html=True)
                    
                    # Keyword suggestions
                    st.markdown("### Keywords and Content")
                    for suggestion in suggestions['keywords']:
                        st.markdown(f'<div class="suggestion-card">{suggestion}</div>', unsafe_allow_html=True)
                    
                    # Contact information suggestions
                    st.markdown("### Contact Information")
                    for suggestion in suggestions['contact']:
                        st.markdown(f'<div class="suggestion-card">{suggestion}</div>', unsafe_allow_html=True)
                    
                    # Readability suggestions
                    st.markdown("### Readability and Style")
                    for suggestion in suggestions['readability']:
                        st.markdown(f'<div class="suggestion-card">{suggestion}</div>', unsafe_allow_html=True)
                    
                    # Industry-specific suggestions
                    st.markdown("### Industry-Specific Suggestions")
                    for suggestion in suggestions['industry']:
                        st.markdown(f'<div class="suggestion-card">{suggestion}</div>', unsafe_allow_html=True)
                    
                    # Achievement suggestions
                    st.markdown("### Achievement Analysis")
                    for suggestion in suggestions['achievements']:
                        st.markdown(f'<div class="suggestion-card">{suggestion}</div>', unsafe_allow_html=True)
                    
                    # Role-specific skill suggestions
                    if 'skills' in suggestions:
                        st.markdown("### Role-Specific Skill Suggestions")
                        for suggestion in suggestions['skills']:
                            st.markdown(f'<div class="suggestion-card">{suggestion}</div>', unsafe_allow_html=True)
                
                with tab3:
                    if job_description:
                        # Process job description
                        job_text = clean_text(job_description)
                        job_keywords = extract_keywords(job_text)
                        
                        # Calculate similarity
                        match_score = calculate_similarity(resume_keywords, job_keywords)
                        
                        # Find matching and missing keywords
                        matching_keywords = set(resume_keywords) & set(job_keywords)
                        missing_keywords = set(job_keywords) - set(resume_keywords)
                        
                        # Display job match results
                        st.markdown("### Job Match Analysis")
                        st.plotly_chart(create_gauge_chart(match_score, "Job Match Score"), use_container_width=True)
                        
                        # Keyword analysis chart
                        st.plotly_chart(create_keyword_chart(matching_keywords, missing_keywords), use_container_width=True)
                        
                        # Matching keywords
                        st.markdown("### Matching Keywords")
                        st.markdown('<div class="suggestion-card">' + 
                                  ', '.join(f'<span class="highlight">{kw}</span>' for kw in matching_keywords) + 
                                  '</div>', unsafe_allow_html=True)
                        
                        # Missing keywords
                        st.markdown("### Missing Keywords")
                        st.markdown('<div class="suggestion-card">' + 
                                  ', '.join(f'<span class="highlight">{kw}</span>' for kw in missing_keywords) + 
                                  '</div>', unsafe_allow_html=True)
                        
                        # Suggestions
                        st.markdown("### Job-Specific Suggestions")
                        if missing_keywords:
                            st.markdown('<div class="suggestion-card">Consider adding these keywords to your resume:</div>', 
                                      unsafe_allow_html=True)
                            for keyword in list(missing_keywords)[:5]:
                                st.markdown(f'<div class="suggestion-card">- {keyword}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="suggestion-card">Great job! Your resume matches well with the job description.</div>', 
                                      unsafe_allow_html=True)
                
                # Best practices in an expander
                with st.expander("💡 Best Practices", expanded=False):
                    st.markdown("""
                    - Keep your resume to 1-2 pages
                    - Use standard fonts (Arial, Calibri, Times New Roman)
                    - Save as PDF for best compatibility
                    - Use keywords from the job description
                    - Quantify achievements with numbers
                    - Use consistent formatting throughout
                    - Avoid tables, images, and complex formatting
                    - Use the STAR method for describing achievements
                    - Include both technical and soft skills
                    - Keep bullet points concise and impactful
                    """)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.write("Please make sure your file is not corrupted and try again.")

if __name__ == "__main__":
    main() 