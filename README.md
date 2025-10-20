# QuickRead AI - Smart PDF Analyzer

A streamlined web application that processes uploaded PDFs to deliver instant textual analysis, including AI-powered summarization and sentiment detection.

## Overview

QuickRead AI is an intelligent PDF analysis tool built with Python and Streamlit. It extracts text from PDF documents and provides:
- **Smart Summarization** - AI-powered document summaries
- **Sentiment Analysis** - Emotional tone detection  
- **Key Points Extraction** - Main ideas at a glance
- **Text Statistics** - Word count, readability metrics
- **PDF Export** - Download summaries as professional reports
- ## 🌐 Live Demo

Try it now: [https://quickread.streamlit.app/](https://quickread.streamlit.app/)

## Tech Stack

- **Python** - Core programming language
- **Streamlit** - Web framework for rapid development
- **NLTK & TextBlob** - Natural Language Processing
- **PyPDF2 & pdfplumber** - PDF text extraction
- **FPDF** - PDF report generation

## Installation

1. **Clone and setup**
```bash
git clone <repository-url>
cd QuickRead
python -m venv venv
source venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt
Run the application

bash
streamlit run app.py
Open browser → http://localhost:8501

How to Use
Upload a PDF file using the sidebar

Select analysis options (summary, sentiment, key points)

Click "Analyze Document"

View instant results and download PDF summary

Project Structure
text
QuickRead/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── README.md          # Documentation
└── utils/
    ├── pdf_processor.py    # PDF text extraction
    └── nlp_analyzer.py     # AI analysis engine
Key Features
Smart PDF Processing
Dual extraction methods for reliability

Handles various PDF formats

Automatic error recovery

Intelligent Analysis
Document type detection (academic, resume, technical)

Context-aware summarization

Accurate sentiment scoring

Professional Output
Clean, readable summaries

Downloadable PDF reports

Structured insights

Use Cases
Academic - Research paper summaries, assignment analysis

Professional - Business reports, resume processing

Personal - Document organization, learning aid

Privacy
Local processing only

No data stored or sent externally

Secure file handling

Contributing
Contributions welcome! Feel free to submit issues and pull requests.

License
MIT License - see LICENSE file for details.
