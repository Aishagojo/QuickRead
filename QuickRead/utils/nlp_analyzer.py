# utils/nlp_analyzer.py
import nltk
from textblob import TextBlob
from typing import Dict, List
import re

class NLPAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # Determine sentiment label
        if sentiment.polarity > 0.1:
            label = "Positive"
        elif sentiment.polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
        
        return {
            "polarity": round(sentiment.polarity, 3),
            "subjectivity": round(sentiment.subjectivity, 3),
            "label": label
        }
    
    def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """Extract key sentences as main points"""
        sentences = nltk.sent_tokenize(text)
        
        # Simple heuristic: longer sentences often contain more information
        scored_sentences = []
        for sentence in sentences:
            # Score by length and presence of important indicators
            score = len(sentence.split())
            if any(keyword in sentence.lower() for keyword in ['important', 'key', 'critical', 'essential', 'major']):
                score += 10
            scored_sentences.append((score, sentence))
        
        # Get top sentences
        scored_sentences.sort(reverse=True)
        key_points = [sentence for _, sentence in scored_sentences[:num_points]]
        
        return key_points
    
    def generate_summary(self, text: str, summary_ratio: float = 0.4) -> str:
        """Generate intelligent summary using advanced text analysis"""
        # First, clean and preprocess the text
        text = self.clean_text(text)
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= 3:
            return text
        
        # Classify document type
        document_type = self._classify_document_type(text)
        
        if document_type == "academic_qna":
            return self._summarize_academic_qna(sentences)
        elif document_type == "resume":
            return self._summarize_resume(sentences)
        elif document_type == "technical":
            return self._summarize_technical(sentences)
        else:
            return self._summarize_general(sentences, summary_ratio)
    
    def _classify_document_type(self, text: str) -> str:
        """Classify the type of document for better summarization"""
        text_lower = text.lower()
        
        # Check for academic/Q&A format
        qna_indicators = ['question:', 'answer:', 'q:', 'a:', 'explain', 'define', 'describe']
        if any(indicator in text_lower for indicator in qna_indicators):
            return "academic_qna"
        
        # Check for resume/CV format
        resume_indicators = ['resume', 'cv', 'experience', 'education', 'skills', 'objective']
        if any(indicator in text_lower for indicator in resume_indicators):
            return "resume"
        
        # Check for technical/problem statement documents
        tech_indicators = ['problem statement', 'challenge', 'overview', 'solution', 'implementation',
                          'blockchain', 'ai', 'knowledge graph', 'chatbot', 'technical', 'algorithm']
        if any(indicator in text_lower for indicator in tech_indicators):
            return "technical"
        
        return "general"
    
    def _summarize_academic_qna(self, sentences: List[str]) -> str:
        """Summarize academic Q&A documents by extracting key concepts"""
        questions = []
        answers = []
        current_question = None
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Detect questions
            if any(sentence_lower.startswith(prefix) for prefix in ['question:', 'q:', 'explain', 'define', 'describe', 'what', 'how', 'why']):
                current_question = sentence
                questions.append(sentence)
            # Detect answers
            elif any(sentence_lower.startswith(prefix) for prefix in ['answer:', 'a:']) and current_question:
                answers.append(sentence)
            # Continue answer for current question
            elif current_question and len(answers) > 0 and answers[-1].startswith('Answer:'):
                # This is continuation of the last answer
                pass
        
        # Create summary by combining key questions and answers
        summary_parts = []
        
        # Add the main topic if identifiable
        if questions:
            main_topic = self._extract_main_topic(questions[0])
            if main_topic:
                summary_parts.append(f"Main Topic: {main_topic}")
        
        # Add key questions and their core answers
        for i, (question, answer) in enumerate(zip(questions[:3], answers[:3])):
            if i < len(answers):
                # Extract the core answer (remove "Answer:" prefix and get to the point)
                clean_answer = re.sub(r'^(Answer:\s*|A:\s*)', '', answer, flags=re.IGNORECASE)
                # Take first sentence of answer as summary
                answer_sentences = nltk.sent_tokenize(clean_answer)
                if answer_sentences:
                    core_answer = answer_sentences[0]
                    summary_parts.append(f"• {question}\n  {core_answer}")
        
        if summary_parts:
            return "\n\n".join(summary_parts)
        else:
            # Fallback: return first few sentences that seem important
            important_sentences = [s for s in sentences if len(s.split()) > 8][:4]
            return " ".join(important_sentences)
    
    def _summarize_resume(self, sentences: List[str]) -> str:
        """Summarize resume documents by extracting key sections"""
        sections = {
            'contact': [],
            'summary': [],
            'education': [],
            'experience': [],
            'skills': []
        }
        
        current_section = None
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Detect section headers
            if any(keyword in sentence_lower for keyword in ['phone', 'email', 'contact']):
                current_section = 'contact'
            elif any(keyword in sentence_lower for keyword in ['summary', 'objective', 'profile']):
                current_section = 'summary'
            elif any(keyword in sentence_lower for keyword in ['education', 'university', 'degree']):
                current_section = 'education'
            elif any(keyword in sentence_lower for keyword in ['experience', 'work', 'employment']):
                current_section = 'experience'
            elif any(keyword in sentence_lower for keyword in ['skills', 'technical', 'programming']):
                current_section = 'skills'
            
            # Add content to current section
            if current_section and len(sentence.strip()) > 10:
                sections[current_section].append(sentence)
        
        # Build summary from key sections
        summary_parts = []
        
        if sections['summary']:
            summary_parts.append("Professional Summary:\n" + " ".join(sections['summary'][:2]))
        
        if sections['experience']:
            summary_parts.append("Key Experience:\n" + " • ".join(sections['experience'][:3]))
        
        if sections['education']:
            summary_parts.append("Education:\n" + " • ".join(sections['education'][:2]))
        
        if sections['skills']:
            skills_text = " ".join(sections['skills'][:5])
            summary_parts.append("Key Skills: " + skills_text)
        
        if summary_parts:
            return "\n\n".join(summary_parts)
        else:
            return " ".join(sentences[:5])
    
    def _summarize_technical(self, sentences: List[str]) -> str:
        """Summarize technical documents and problem statements"""
        sections = {
            'overview': [],
            'problem': [],
            'challenge': [],
            'solution': [],
            'implementation': []
        }
        
        current_section = 'overview'
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Detect section headers in technical documents
            if any(keyword in sentence_lower for keyword in ['overview', 'introduction', 'background']):
                current_section = 'overview'
            elif any(keyword in sentence_lower for keyword in ['problem', 'issue', 'limitation']):
                current_section = 'problem'
            elif any(keyword in sentence_lower for keyword in ['challenge', 'objective', 'goal']):
                current_section = 'challenge'
            elif any(keyword in sentence_lower for keyword in ['solution', 'approach', 'method']):
                current_section = 'solution'
            elif any(keyword in sentence_lower for keyword in ['implementation', 'technical', 'code', 'algorithm']):
                current_section = 'implementation'
            
            # Add meaningful content to current section
            if len(sentence.strip()) > 15 and not sentence.strip().startswith(';'):
                sections[current_section].append(sentence)
        
        # Build comprehensive summary
        summary_parts = []
        
        if sections['overview']:
            # Take the most important overview sentences
            overview_summary = " ".join(sections['overview'][:2])
            summary_parts.append(f"Overview: {overview_summary}")
        
        if sections['problem']:
            problem_summary = " ".join(sections['problem'][:2])
            summary_parts.append(f"Problem: {problem_summary}")
        
        if sections['challenge']:
            challenge_summary = " ".join(sections['challenge'][:2])
            summary_parts.append(f"Challenge: {challenge_summary}")
        
        if sections['solution']:
            solution_summary = " ".join(sections['solution'][:2])
            summary_parts.append(f"Solution Approach: {solution_summary}")
        
        # If we don't have enough structured content, fall back to intelligent extraction
        if len(summary_parts) < 2:
            return self._summarize_technical_fallback(sentences)
        
        return "\n\n".join(summary_parts)
    
    def _summarize_technical_fallback(self, sentences: List[str]) -> str:
        """Fallback method for technical document summarization"""
        # Extract key sentences that contain important technical concepts
        important_sentences = []
        technical_keywords = [
            'develop', 'create', 'build', 'implement', 'design', 'architecture',
            'integration', 'system', 'model', 'framework', 'platform', 'solution',
            'challenge', 'problem', 'objective', 'goal', 'purpose'
        ]
        
        for sentence in sentences:
            if len(sentence.split()) < 5:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            
            # Score based on technical importance
            score = 0
            
            # High importance for sentences with technical keywords
            for keyword in technical_keywords:
                if keyword in sentence_lower:
                    score += 3
            
            # Higher importance for sentences that define or describe
            if any(word in sentence_lower for word in ['is a', 'are', 'means', 'defines', 'describes']):
                score += 2
            
            # Avoid code examples and configuration lines
            if sentence.strip().startswith(';') or 'match self' in sentence_lower:
                score = -10
            
            if score > 0:
                important_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        important_sentences.sort(key=lambda x: x[0], reverse=True)
        
        if important_sentences:
            # Take top 4-6 sentences
            top_sentences = [sentence for score, sentence in important_sentences[:6]]
            return " ".join(top_sentences)
        else:
            # Final fallback: take first meaningful sentences
            meaningful = [s for s in sentences if len(s.split()) > 8][:5]
            return " ".join(meaningful)
    
    def _summarize_general(self, sentences: List[str], summary_ratio: float) -> str:
        """Summarize general documents using intelligent scoring"""
        # Score sentences based on multiple factors
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 5:  # Skip very short sentences
                continue
                
            score = 0
            
            # Position scoring
            if i == 0:  # First sentence often contains main idea
                score += 5
            elif i < 3:  # Early sentences
                score += 2
            elif i == len(sentences) - 1:  # Last sentence (conclusion)
                score += 3
            
            # Length scoring (medium length sentences are often most informative)
            word_count = len(sentence.split())
            if 12 <= word_count <= 25:
                score += 3
            elif word_count > 25:
                score += 1
            
            # Keyword scoring
            important_keywords = [
                'conclusion', 'summary', 'important', 'key', 'main', 'primary',
                'result', 'finding', 'study', 'research', 'analysis', 'purpose'
            ]
            
            sentence_lower = sentence.lower()
            for keyword in important_keywords:
                if keyword in sentence_lower:
                    score += 3
            
            # Question indicators (in academic texts)
            if any(word in sentence_lower for word in ['question', 'explain', 'define', 'describe']):
                score += 2
            
            scored_sentences.append((score, sentence, i))
        
        if not scored_sentences:
            return " ".join(sentences[:4])
        
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        num_sentences = max(3, min(6, int(len(sentences) * summary_ratio)))
        top_sentences = [sentence for score, sentence, idx in scored_sentences[:num_sentences]]
        
        # Reorder to maintain coherence
        top_indices = [idx for score, sentence, idx in scored_sentences[:num_sentences]]
        top_sentences_ordered = [sentence for _, sentence in sorted(zip(top_indices, top_sentences))]
        
        return " ".join(top_sentences_ordered)
    
    def _extract_main_topic(self, text: str) -> str:
        """Extract main topic from text"""
        # Look for technical terms and key concepts
        technical_terms = ['router', 'switch', 'ipv4', 'ipv6', 'network', 'protocol', 
                          'algorithm', 'database', 'programming', 'security']
        
        words = text.lower().split()
        for term in technical_terms:
            if term in words:
                return term.capitalize()
        
        return ""

    def extract_structured_info(self, text: str) -> Dict:
        """Extract structured information from resume-like text"""
        lines = text.split('\n')
        structured_data = {
            'education': [],
            'experience': [],
            'skills': [],
            'projects': []
        }
        
        current_section = None
        section_keywords = {
            'education': ['education', 'university', 'college', 'degree'],
            'experience': ['experience', 'intern', 'work', 'employment'],
            'skills': ['skills', 'technical', 'programming', 'languages'],
            'projects': ['projects', 'portfolio', 'github']
        }
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line starts a new section
            for section, keywords in section_keywords.items():
                if any(keyword in line_lower for keyword in keywords) and len(line.split()) < 5:
                    current_section = section
                    break
            
            # Add content to current section
            if current_section and line.strip() and len(line.strip()) > 10:
                if line not in structured_data[current_section]:
                    structured_data[current_section].append(line.strip())
        
        return structured_data
    
    def get_text_stats(self, text: str) -> Dict:
        """Get basic text statistics"""
        words = text.split()
        sentences = nltk.sent_tokenize(text)
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "character_count": len(text),
            "avg_word_length": round(sum(len(word) for word in words) / len(words), 1) if words else 0,
            "avg_sentence_length": round(len(words) / len(sentences), 1) if sentences else 0
        }

# Test function
def test_nlp_analyzer():
    analyzer = NLPAnalyzer()
    sample_text = "This is a wonderful product! I love using it every day. The quality is exceptional and the service is great."
    
    sentiment = analyzer.analyze_sentiment(sample_text)
    stats = analyzer.get_text_stats(sample_text)
    
    print("NLP Analyzer initialized successfully!")
    print(f"Sentiment: {sentiment}")
    print(f"Stats: {stats}")