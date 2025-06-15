#!/usr/bin/env python3
"""
🤖 ULTIMATE AI SOCIAL MEDIA AGENT - JUSTEMMA EDITION
Complete autonomous system for social media growth and lead generation
"""

import os
import json
import asyncio
import aiohttp
import tweepy
import praw
import time
import random
import schedule
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import sqlite3
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content, HtmlContent
import requests
from bs4 import BeautifulSoup
import openai
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
from transformers import pipeline
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import threading
import queue
from collections import defaultdict
import re
import hashlib
from urllib.parse import urljoin, urlparse
import yfinance as yf
from github import Github
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from dotenv import load_dotenv
from flask import Flask, jsonify
import gunicorn.app.base

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Global initialization status
app.initialization_status = {
    'status': 'initializing',
    'start_time': datetime.now().isoformat(),
    'agent_ready': False
}

@app.route('/health')
def health_check():
    """Health check endpoint that works even during initialization"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'initialization_status': app.initialization_status['status'],
        'agent_ready': app.initialization_status['agent_ready']
    })

@app.route('/startup-status')
def startup_status():
    """Detailed startup status endpoint"""
    return jsonify({
        'status': app.initialization_status['status'],
        'start_time': app.initialization_status['start_time'],
        'agent_ready': app.initialization_status['agent_ready'],
        'uptime_seconds': (datetime.now() - datetime.fromisoformat(app.initialization_status['start_time'])).total_seconds()
    })

# Status endpoint
@app.route('/status')
def status():
    """Status endpoint that provides detailed agent status and analytics"""
    try:
        # Check initialization status
        if not hasattr(app, 'agent'):
            return jsonify({
                'status': 'initializing',
                'initialization_status': app.initialization_status['status'],
                'agent_ready': app.initialization_status['agent_ready'],
                'start_time': app.initialization_status['start_time'],
                'uptime_seconds': (datetime.now() - datetime.fromisoformat(app.initialization_status['start_time'])).total_seconds(),
                'message': 'Agent is still initializing. Please check back in a few moments.'
            }), 200
        
        # Get analytics if agent is ready
        analytics = app.agent.get_daily_analytics()
        return jsonify({
            'status': 'running',
            'initialization_status': app.initialization_status['status'],
            'agent_ready': app.initialization_status['agent_ready'],
            'start_time': app.initialization_status['start_time'],
            'uptime_seconds': (datetime.now() - datetime.fromisoformat(app.initialization_status['start_time'])).total_seconds(),
            'analytics': analytics,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'initialization_status': app.initialization_status['status'],
            'agent_ready': app.initialization_status['agent_ready'],
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/')
def index():
    """Root endpoint that provides basic service information"""
    return jsonify({
        'service': 'JUSTEMMA AI Social Media Agent',
        'status': 'operational',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'status': '/status'
        },
        'description': 'An autonomous AI agent for social media management, lead generation, and business engagement',
        'features': [
            'Autonomous posting (8 posts/day)',
            'Smart community engagement',
            'Lead generation & scoring',
            'Trend monitoring & analysis',
            'Competitor analysis',
            'Performance optimization',
            'Daily reports',
            'Visual content creation'
        ]
    })

# Custom Gunicorn application
class StandaloneApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            self.cfg.set(key, value)

    def load(self):
        return self.application

@dataclass
class Config:
    # Twitter API
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
    TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET')
    TWITTER_CLIENT_ID = os.getenv('TWITTER_CLIENT_ID')
    TWITTER_CLIENT_SECRET = os.getenv('TWITTER_CLIENT_SECRET')
    
    # Reddit API
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
    
    # AI APIs
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    HF_TOKEN = os.getenv('HF_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Email (Optional)
    SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
    EMAIL_USER = os.getenv('EMAIL_USER')
    REPORT_EMAIL = os.getenv('REPORT_EMAIL')
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    EMAIL_FROM = os.getenv('EMAIL_FROM')
    EMAIL_TO = os.getenv('EMAIL_TO')
    
    # Database
    DB_NAME = os.getenv('DB_NAME', 'aiagent.db')
    
    # Posting Schedule
    DAILY_POSTS = int(os.getenv('DAILY_POSTS', '8'))
    ENGAGEMENT_LIMIT = int(os.getenv('ENGAGEMENT_LIMIT', '200'))
    FOLLOW_LIMIT = int(os.getenv('FOLLOW_LIMIT', '50'))
    
    # Business Categories for Lead Generation
    BUSINESS_CATEGORIES = [
        "retail", "ecommerce", "restaurant", "grocery", "boutique",
        "salon", "spa", "fitness", "coffee_shop", "bakery",
        "clothing_store", "jewelry_store", "bookstore", "pharmacy",
        "pet_shop", "hardware_store", "furniture_store", "gift_shop"
    ]
    
    # Content Categories
    CONTENT_CATEGORIES = [
        "business_website_tips", "ecommerce_success", "online_presence",
        "digital_marketing", "business_growth", "customer_engagement",
        "local_business", "retail_tech", "business_automation"
    ]
    
    def __post_init__(self):
        """Validate required API keys"""
        # Core required keys
        required_keys = {
            'OPENAI_API_KEY': self.OPENAI_API_KEY,
            'TWITTER_API_KEY': self.TWITTER_API_KEY,
            'TWITTER_API_SECRET': self.TWITTER_API_SECRET,
            'TWITTER_BEARER_TOKEN': self.TWITTER_BEARER_TOKEN,
            'TWITTER_ACCESS_TOKEN': self.TWITTER_ACCESS_TOKEN,
            'TWITTER_ACCESS_SECRET': self.TWITTER_ACCESS_SECRET,
            'REDDIT_CLIENT_ID': self.REDDIT_CLIENT_ID,
            'REDDIT_CLIENT_SECRET': self.REDDIT_CLIENT_SECRET,
            'GEMINI_API_KEY': self.GEMINI_API_KEY,
            'HF_TOKEN': self.HF_TOKEN,
        }
        
        # Email-related keys (SendGrid only)
        email_keys = {
            'SENDGRID_API_KEY': self.SENDGRID_API_KEY,
            'EMAIL_FROM': self.EMAIL_FROM,
            'EMAIL_TO': self.EMAIL_TO,
            'REPORT_EMAIL': self.REPORT_EMAIL
        }
        
        # Check if SendGrid is configured
        has_sendgrid = all([self.SENDGRID_API_KEY, self.EMAIL_FROM, self.EMAIL_TO])
        if not has_sendgrid:
            print("[WARNING] SendGrid email functionality will be disabled. Missing required SendGrid variables")
            # Clear email-related variables to disable email functionality
            self.EMAIL_FROM = None
            self.EMAIL_TO = None
            self.SENDGRID_API_KEY = None
            self.REPORT_EMAIL = None
        
        # Check core required keys
        missing_keys = [key for key, value in required_keys.items() if not value]
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")

class PostType(Enum):
    GREETING = "greeting"
    BUSINESS_TIP = "business_tip"
    SUCCESS_STORY = "success_story"
    ECOMMERCE_TIP = "ecommerce_tip"
    DIGITAL_MARKETING = "digital_marketing"
    BUSINESS_GROWTH = "business_growth"
    CUSTOMER_ENGAGEMENT = "customer_engagement"
    LOCAL_BUSINESS = "local_business"
    RETAIL_TECH = "retail_tech"

class SocialMediaAgent:
    PostType = PostType  # Add this line to make PostType accessible as a class variable
    
    def __init__(self):
        self.config = Config()
        self.setup_logging()
        self.setup_database()
        self.setup_apis()
        self.setup_ai_models()
        self.content_queue = queue.Queue()
        self.engagement_queue = queue.Queue()
        self.lead_queue = queue.Queue()
        self.is_running = False
        self.first_run = True
        
        # Initialize SendGrid client
        self.sg = SendGridAPIClient(self.config.SENDGRID_API_KEY)
        
        # Send startup notification
        self.send_startup_notification()
    
    def send_startup_notification(self):
        """Send notification email when agent starts using SendGrid"""
        try:
            subject = "🚀 JustEmma AI Agent Started Successfully"
            html_content = f"""
            <html>
            <body>
            <h2>🤖 JustEmma AI Agent is now running!</h2>
            
            <p>The AI agent has been initialized with the following features:</p>
            <ul>
                <li>✅ Autonomous posting (8 posts/day)</li>
                <li>✅ Smart community engagement</li>
                <li>✅ Lead generation & scoring</li>
                <li>✅ Trend monitoring & analysis</li>
                <li>✅ Competitor analysis</li>
                <li>✅ Performance optimization</li>
                <li>✅ Daily reports</li>
                <li>✅ Visual content creation</li>
            </ul>
            
            <p><strong>Startup Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Environment:</strong> {os.getenv('ENVIRONMENT', 'Production')}</p>
            
            <p>You will receive daily performance reports at the end of each day.</p>
            
            <p><em>This is an automated message from JustEmma AI Agent</em></p>
            </body>
            </html>
            """
            
            self.send_email(subject, html_content)
            self.logger.info("Startup notification email sent successfully")
            
        except Exception as e:
            self.logger.error(f"Error sending startup notification: {e}")
    
    def send_email(self, subject: str, html_content: str):
        """Send email using SendGrid"""
        try:
            message = Mail(
                from_email=Email(self.config.EMAIL_FROM),
                to_emails=To(self.config.EMAIL_TO),
                subject=subject,
                html_content=HtmlContent(html_content)
            )
            
            # Send email
            response = self.sg.send(message)
            
            # Log success
            self.logger.info(f"Email sent successfully: {subject} (Status: {response.status_code})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email via SendGrid: {e}")
            return False
    
    def send_email_report(self, report: str):
        """Send daily report via SendGrid"""
        try:
            subject = f"📊 JustEmma AI Agent Daily Report - {datetime.now().strftime('%Y-%m-%d')}"
            self.send_email(subject, report)
            
        except Exception as e:
            self.logger.error(f"Error sending daily report: {e}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging with UTF-8 encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                # File handler with UTF-8 encoding
                logging.FileHandler('logs/ai_agent.log', encoding='utf-8'),
                # Stream handler with UTF-8 encoding for console
                logging.StreamHandler(open(1, 'w', encoding='utf-8', closefd=False))
            ]
        )
        
        # Create logger
        self.logger = logging.getLogger(__name__)
        
        # Add a filter to handle emoji characters
        class EmojiFilter(logging.Filter):
            def filter(self, record):
                # Replace problematic emoji characters with text equivalents
                emoji_map = {
                    '🚀': '[ROCKET]',
                    '✅': '[CHECK]',
                    '🎯': '[TARGET]',
                    '🤖': '[ROBOT]',
                    '👋': '[WAVE]',
                    '💡': '[IDEA]',
                    '🛠️': '[TOOLS]',
                    '🔥': '[FIRE]',
                    '📚': '[BOOKS]',
                    '🤔': '[THINK]',
                    '🌟': '[STAR]'
                }
                if isinstance(record.msg, str):
                    for emoji, text in emoji_map.items():
                        record.msg = record.msg.replace(emoji, text)
                return True
        
        # Add the filter to the logger
        self.logger.addFilter(EmojiFilter())
    
    def setup_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect(self.config.DB_NAME, check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Posts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                post_type TEXT NOT NULL,
                platform TEXT NOT NULL,
                post_id TEXT,
                engagement_score INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hashtags TEXT,
                mentions INTEGER DEFAULT 0,
                retweets INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0
            )
        ''')
        
        # Leads table with updated schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                platform TEXT NOT NULL,
                profile_url TEXT,
                business_type TEXT,
                pain_points TEXT,
                contact_info TEXT,
                engagement_level INTEGER DEFAULT 0,
                last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'new',
                dm_status TEXT DEFAULT 'not_sent',
                dm_sent_at TIMESTAMP,
                dm_response TEXT,
                follow_up_count INTEGER DEFAULT 0,
                last_follow_up TIMESTAMP
            )
        ''')
        
        # DMs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS direct_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response TEXT,
                response_at TIMESTAMP,
                status TEXT DEFAULT 'sent',
                FOREIGN KEY (lead_id) REFERENCES leads(id)
            )
        ''')
        
        # Analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                date_recorded TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                platform TEXT NOT NULL
            )
        ''')
        
        # Trends table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trend_topic TEXT NOT NULL,
                trend_score REAL NOT NULL,
                platform TEXT NOT NULL,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        self.conn.commit()
    
    def setup_apis(self):
        """Initialize all API connections"""
        try:
            # Twitter API v2
            self.twitter_client = tweepy.Client(
                bearer_token=self.config.TWITTER_BEARER_TOKEN,
                consumer_key=self.config.TWITTER_API_KEY,
                consumer_secret=self.config.TWITTER_API_SECRET,
                access_token=self.config.TWITTER_ACCESS_TOKEN,
                access_token_secret=self.config.TWITTER_ACCESS_SECRET,
                wait_on_rate_limit=True
            )
            
            # Reddit API
            self.reddit = praw.Reddit(
                client_id=self.config.REDDIT_CLIENT_ID,
                client_secret=self.config.REDDIT_CLIENT_SECRET,
                user_agent=self.config.REDDIT_USER_AGENT
            )
            
            # Gemini API
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            
            self.logger.info("All APIs initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up APIs: {e}")
    
    def setup_ai_models(self):
        """Initialize AI models and agents"""
        try:
            # LangChain setup with updated import
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=self.config.OPENAI_API_KEY
            )
            
            # Sentiment analyzer
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # HuggingFace models
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            self.classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            self.logger.info("AI models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up AI models: {e}")
    
    def generate_content(self, post_type: PostType, context: Dict = None) -> str:
        """Generate engaging content focused on business websites and e-commerce"""
        try:
            if post_type == PostType.GREETING and self.first_run:
                timestamp = datetime.now().strftime("%H:%M")
                return f"🚀 Hello from JustEmma! 👋 Helping local businesses and retailers build their online presence at {timestamp}. Let's grow your business together! #SmallBusiness #Ecommerce #WebDevelopment #BusinessGrowth"
            
            # Check for recent similar content
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT content FROM posts 
                WHERE post_type = ? 
                AND created_at > datetime('now', '-1 day')
                ORDER BY created_at DESC
                LIMIT 5
            ''', (post_type.value,))
            recent_posts = [row[0] for row in cursor.fetchall()]
            
            max_attempts = 3
            for attempt in range(max_attempts):
                prompts = {
                    PostType.BUSINESS_TIP: f"Create a unique tip for small businesses about improving their online presence (attempt {attempt + 1}). Focus on website benefits.",
                    PostType.SUCCESS_STORY: f"Share a success story about a local business that improved sales with a website (attempt {attempt + 1}).",
                    PostType.ECOMMERCE_TIP: f"Provide a unique e-commerce tip for small retailers (attempt {attempt + 1}).",
                    PostType.DIGITAL_MARKETING: f"Share insights about digital marketing for local businesses (attempt {attempt + 1}).",
                    PostType.BUSINESS_GROWTH: f"Create content about how a website can help business growth (attempt {attempt + 1}).",
                    PostType.CUSTOMER_ENGAGEMENT: f"Share tips about online customer engagement for businesses (attempt {attempt + 1}).",
                    PostType.LOCAL_BUSINESS: f"Create content about helping local businesses go digital (attempt {attempt + 1}).",
                    PostType.RETAIL_TECH: f"Share insights about technology for retail businesses (attempt {attempt + 1})."
                }
                
                base_prompt = prompts.get(post_type, "Create unique engaging business content")
                
                # Add context if available
                if context:
                    base_prompt += f"\n\nContext: {json.dumps(context)}"
                
                # Add personality and requirements
                full_prompt = f"""
                {base_prompt}
                
                Requirements:
                - Keep it under 280 characters for Twitter
                - Include relevant hashtags (3-5)
                - Make it engaging and valuable
                - Use emojis appropriately
                - Focus on business websites, e-commerce, and online presence
                - Show expertise in helping businesses grow online
                - Encourage engagement
                - Add a unique timestamp or identifier
                
                Write from the perspective of JustEmma, a web development expert helping businesses grow online.
                """
                
                # Generate content using Gemini
                response = self.gemini_model.generate_content(full_prompt)
                content = response.text.strip()
                
                # Add timestamp to ensure uniqueness
                timestamp = datetime.now().strftime("%H:%M")
                content = f"{content} ({timestamp})"
                
                # Check if content is too similar to recent posts
                is_duplicate = False
                for recent_post in recent_posts:
                    if self.calculate_similarity(content, recent_post) > 0.8:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # Add trending hashtags
                    content = self.add_trending_hashtags(content)
                    return content
            
            # If all attempts failed, use fallback with timestamp
            return f"{self.get_fallback_content(post_type)} ({datetime.now().strftime('%H:%M')})"
            
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            return f"{self.get_fallback_content(post_type)} ({datetime.now().strftime('%H:%M')})"
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_fallback_content(self, post_type: PostType) -> str:
        """Fallback content focused on business websites"""
        fallbacks = {
            PostType.GREETING: "🚀 Hello from JustEmma! Helping businesses grow online with beautiful websites! #SmallBusiness #WebDevelopment #BusinessGrowth",
            PostType.BUSINESS_TIP: "💡 Pro tip: A professional website can increase your business credibility by 75%! #BusinessTips #WebDevelopment",
            PostType.SUCCESS_STORY: "🌟 Local bakery increased sales by 40% after launching their website! #SuccessStory #BusinessGrowth",
            PostType.ECOMMERCE_TIP: "🛍️ Make your products available 24/7 with an e-commerce website! #Ecommerce #BusinessTips",
            PostType.DIGITAL_MARKETING: "📱 Your website is your digital storefront. Make it count! #DigitalMarketing #BusinessGrowth",
            PostType.BUSINESS_GROWTH: "📈 85% of customers research online before visiting a store. Is your business ready? #BusinessGrowth",
            PostType.CUSTOMER_ENGAGEMENT: "🤝 Connect with customers 24/7 through your website! #CustomerEngagement #BusinessTips",
            PostType.LOCAL_BUSINESS: "🏪 Local businesses: Your website is your best salesperson! #LocalBusiness #WebDevelopment",
            PostType.RETAIL_TECH: "💻 Modern retail needs a modern website! #RetailTech #BusinessGrowth"
        }
        return fallbacks.get(post_type, "🚀 Helping businesses grow online! #WebDevelopment #BusinessGrowth")
    
    def add_trending_hashtags(self, content: str) -> str:
        """Add trending hashtags to content"""
        trending_hashtags = self.get_trending_hashtags()
        
        # Count existing hashtags
        existing_hashtags = len(re.findall(r'#\w+', content))
        
        # Add trending hashtags if we have space
        if existing_hashtags < 5 and trending_hashtags:
            hashtags_to_add = trending_hashtags[:5-existing_hashtags]
            content += " " + " ".join(f"#{tag}" for tag in hashtags_to_add)
        
        return content
    
    def get_trending_hashtags(self) -> List[str]:
        """Get trending hashtags from database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT trend_topic FROM trends 
            WHERE status = 'active' 
            ORDER BY trend_score DESC 
            LIMIT 10
        ''')
        
        trends = [row[0] for row in cursor.fetchall()]
        return trends if trends else ["WebDevelopment", "AI", "TechInnovation", "Coding", "Programming"]
    
    def post_to_twitter(self, content: str, post_type: PostType) -> Optional[str]:
        """Post content to Twitter"""
        try:
            response = self.twitter_client.create_tweet(text=content)
            tweet_id = response.data['id']
            
            # Save to database
            self.save_post(content, post_type.value, "twitter", tweet_id)
            
            self.logger.info(f"Posted to Twitter: {content[:50]}...")
            return tweet_id
            
        except Exception as e:
            self.logger.error(f"Error posting to Twitter: {e}")
            return None
    
    def save_post(self, content: str, post_type: str, platform: str, post_id: str = None):
        """Save post to database"""
        cursor = self.conn.cursor()
        hashtags = " ".join(re.findall(r'#\w+', content))
        
        cursor.execute('''
            INSERT INTO posts (content, post_type, platform, post_id, hashtags)
            VALUES (?, ?, ?, ?, ?)
        ''', (content, post_type, platform, post_id, hashtags))
        
        self.conn.commit()
    
    def scrape_trends(self):
        """Scrape trending topics from various sources"""
        trends = []
        
        # GitHub trending
        trends.extend(self.scrape_github_trends())
        
        # Reddit trends
        trends.extend(self.scrape_reddit_trends())
        
        # Tech news trends
        trends.extend(self.scrape_tech_news_trends())
        
        # Save trends to database
        self.save_trends(trends)
        
        return trends
    
    def scrape_github_trends(self) -> List[Dict]:
        """Scrape GitHub trending repositories"""
        trends = []
        try:
            url = "https://github.com/trending"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            repos = soup.find_all('article', class_='Box-row')
            for repo in repos[:10]:  # Top 10 trending
                name_elem = repo.find('h2', class_='h3')
                if name_elem:
                    repo_name = name_elem.text.strip().replace('\n', '').replace(' ', '')
                    trends.append({
                        'topic': repo_name.split('/')[-1],
                        'score': 10,  # High score for GitHub trending
                        'source': 'github'
                    })
            
        except Exception as e:
            self.logger.error(f"Error scraping GitHub trends: {e}")
        
        return trends
    
    def scrape_reddit_trends(self) -> List[Dict]:
        """Scrape trending topics from Reddit"""
        trends = []
        try:
            subreddits = ['programming', 'webdev', 'MachineLearning', 'cybersecurity', 'technology']
            
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                hot_posts = subreddit.hot(limit=10)
                
                for post in hot_posts:
                    # Extract keywords from title
                    words = re.findall(r'\b[A-Za-z]{3,}\b', post.title.lower())
                    tech_keywords = [w for w in words if w in ['python', 'javascript', 'react', 'ai', 'ml', 'web', 'api', 'database', 'security']]
                    
                    for keyword in tech_keywords:
                        trends.append({
                            'topic': keyword,
                            'score': post.score / 100,  # Normalize score
                            'source': 'reddit'
                        })
        
        except Exception as e:
            self.logger.error(f"Error scraping Reddit trends: {e}")
        
        return trends
    
    def scrape_tech_news_trends(self) -> List[Dict]:
        """Scrape trending topics from tech news"""
        trends = []
        try:
            # TechCrunch RSS
            feed_url = "https://techcrunch.com/feed/"
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:10]:
                title = entry.title.lower()
                # Extract tech keywords
                tech_keywords = re.findall(r'\b(ai|ml|blockchain|crypto|web3|saas|api|cloud|security|mobile|app)\b', title)
                
                for keyword in tech_keywords:
                    trends.append({
                        'topic': keyword,
                        'score': 8,  # Good score for news trends
                        'source': 'techcrunch'
                    })
        
        except Exception as e:
            self.logger.error(f"Error scraping tech news trends: {e}")
        
        return trends
    
    def save_trends(self, trends: List[Dict]):
        """Save trends to database"""
        cursor = self.conn.cursor()
        
        for trend in trends:
            cursor.execute('''
                INSERT OR REPLACE INTO trends (trend_topic, trend_score, platform)
                VALUES (?, ?, ?)
            ''', (trend['topic'], trend['score'], trend['source']))
        
        self.conn.commit()
    
    def engage_with_community(self):
        """Engage with the developer community"""
        try:
            # Search for relevant tweets
            tweets = self.twitter_client.search_recent_tweets(
                query="web development OR programming OR AI OR machine learning -is:retweet",
                max_results=20,
                tweet_fields=['author_id', 'public_metrics']
            )
            
            if tweets.data:
                for tweet in tweets.data:
                    # Analyze tweet sentiment and relevance
                    if self.should_engage_with_tweet(tweet.text):
                        self.engage_with_tweet(tweet)
                        time.sleep(2)  # Rate limiting
            
        except Exception as e:
            self.logger.error(f"Error engaging with community: {e}")
    
    def should_engage_with_tweet(self, text: str) -> bool:
        """Determine if we should engage with a tweet"""
        # Check for relevant keywords
        relevant_keywords = ['web development', 'programming', 'coding', 'javascript', 'python', 'react', 'ai', 'machine learning']
        
        text_lower = text.lower()
        for keyword in relevant_keywords:
            if keyword in text_lower:
                # Check sentiment - engage with positive/neutral content
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                if sentiment['compound'] >= -0.3:  # Not too negative
                    return True
        
        return False
    
    def engage_with_tweet(self, tweet):
        """Engage with a specific tweet"""
        try:
            # Generate contextual reply
            reply = self.generate_contextual_reply(tweet.text)
            
            if reply:
                # Like the tweet
                self.twitter_client.like(tweet.id)
                
                # Reply to the tweet
                self.twitter_client.create_tweet(text=reply, in_reply_to_tweet_id=tweet.id)
                
                self.logger.info(f"Engaged with tweet: {tweet.text[:50]}...")
                
                # Check if this could be a lead
                self.analyze_potential_lead(tweet)
        
        except Exception as e:
            self.logger.error(f"Error engaging with tweet: {e}")
    
    def generate_contextual_reply(self, tweet_text: str) -> str:
        """Generate a contextual reply to a tweet"""
        try:
            prompt = f"""
            Generate a helpful, engaging reply to this tweet: "{tweet_text}"
            
            Requirements:
            - Keep it under 280 characters
            - Be helpful and add value
            - Show expertise in web development/AI
            - Be friendly and professional
            - Don't be salesy or promotional
            - Include relevant emoji if appropriate
            """
            
            response = self.gemini_model.generate_content(prompt)
            reply = response.text.strip()
            
            # Ensure it's not too long
            if len(reply) > 280:
                reply = reply[:277] + "..."
            
            return reply
            
        except Exception as e:
            self.logger.error(f"Error generating reply: {e}")
            return None
    
    def analyze_potential_lead(self, tweet):
        """Analyze if a tweet author could be a potential lead"""
        try:
            # Get user info
            user = self.twitter_client.get_user(id=tweet.author_id, user_fields=['description', 'location', 'public_metrics'])
            
            if user.data:
                bio = user.data.description or ""
                
                # Check for business indicators
                business_keywords = ['founder', 'ceo', 'startup', 'business', 'company', 'entrepreneur', 'freelancer']
                
                for keyword in business_keywords:
                    if keyword.lower() in bio.lower():
                        # Potential lead - save to database
                        self.save_lead(user.data)
                        break
        
        except Exception as e:
            self.logger.error(f"Error analyzing potential lead: {e}")
    
    def save_lead(self, user_data):
        """Save potential lead to database"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO leads (username, platform, profile_url, business_type, contact_info)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_data.username,
            'twitter',
            f"https://twitter.com/{user_data.username}",
            user_data.description,
            user_data.description
        ))
        
        self.conn.commit()
        self.logger.info(f"Saved potential lead: {user_data.username}")
    
    def analyze_user_pain_points(self, username: str) -> List[str]:
        """Analyze user's profile and tweets to identify specific pain points"""
        try:
            pain_points = []
            
            # Get user info and recent tweets
            user = self.twitter_client.get_user(username=username, user_fields=['description', 'public_metrics'])
            if not user.data:
                return ['limited online presence', 'manual operations', 'local customer base only']
            
            tweets = self.twitter_client.get_users_tweets(
                user.data.id,
                max_results=20,  # Analyze last 20 tweets
                tweet_fields=['created_at', 'public_metrics']
            )
            
            # Analyze bio for business type and pain points
            bio = user.data.description.lower()
            
            # Common pain point indicators
            indicators = {
                'time': ['busy', 'no time', 'overwhelmed', 'stress', 'rush', 'hectic', 'working late'],
                'customers': ['no customers', 'slow business', 'quiet', 'empty', 'slow day', 'need customers'],
                'online': ['no website', 'need website', 'online presence', 'social media', 'digital', 'online store'],
                'operations': ['manual', 'paperwork', 'inventory', 'stock', 'orders', 'bookings', 'appointments'],
                'marketing': ['advertising', 'marketing', 'promotion', 'reach', 'visibility', 'awareness'],
                'tech': ['tech', 'system', 'software', 'app', 'digital', 'automation', 'online', 'website'],
                'growth': ['grow', 'expansion', 'scale', 'bigger', 'more sales', 'increase revenue']
            }
            
            # Analyze bio for pain points
            for category, words in indicators.items():
                if any(word in bio for word in words):
                    if category == 'time':
                        pain_points.append('manual operations taking too much time')
                    elif category == 'customers':
                        pain_points.append('limited to local customer base only')
                    elif category == 'online':
                        pain_points.append('no online presence affecting growth')
                    elif category == 'operations':
                        pain_points.append('manual business processes causing inefficiencies')
                    elif category == 'marketing':
                        pain_points.append('limited marketing reach affecting growth')
                    elif category == 'tech':
                        pain_points.append('lack of digital tools affecting efficiency')
                    elif category == 'growth':
                        pain_points.append('business growth limited by current systems')
            
            # Analyze tweets for additional pain points
            if tweets.data:
                tweet_texts = [tweet.text.lower() for tweet in tweets.data]
                for category, words in indicators.items():
                    if any(word in ' '.join(tweet_texts) for word in words):
                        if category not in [p.split()[0] for p in pain_points]:
                            if category == 'time':
                                pain_points.append('time-consuming manual processes')
                            elif category == 'customers':
                                pain_points.append('difficulty reaching new customers')
                            elif category == 'online':
                                pain_points.append('missing online sales opportunities')
                            elif category == 'operations':
                                pain_points.append('inefficient business operations')
                            elif category == 'marketing':
                                pain_points.append('limited marketing capabilities')
                            elif category == 'tech':
                                pain_points.append('lack of digital solutions')
                            elif category == 'growth':
                                pain_points.append('growth limited by current setup')
            
            # If no specific pain points found, use business type defaults
            if not pain_points:
                business_type = self.identify_business_type(bio)
                pain_points = self.identify_pain_points(business_type)
            
            # Remove duplicates and limit to top 3 most relevant
            unique_pain_points = list(set(pain_points))
            return unique_pain_points[:3]
            
        except Exception as e:
            self.logger.error(f"Error analyzing user pain points: {e}")
            return ['limited online presence', 'manual operations', 'local customer base only']

    def identify_business_type(self, bio: str) -> str:
        """Identify business type from bio text"""
        business_types = {
            'retail': ['retail', 'store', 'shop', 'boutique'],
            'restaurant': ['restaurant', 'cafe', 'coffee shop', 'food', 'dining'],
            'grocery': ['grocery', 'market', 'supermarket', 'food store'],
            'salon': ['salon', 'spa', 'beauty', 'hair', 'stylist'],
            'boutique': ['boutique', 'fashion', 'clothing', 'apparel'],
            'coffee_shop': ['coffee', 'cafe', 'coffee shop', 'coffeehouse'],
            'bakery': ['bakery', 'bake', 'pastry', 'bread'],
            'clothing_store': ['clothing', 'fashion', 'apparel', 'wear'],
            'jewelry_store': ['jewelry', 'jewellery', 'accessories'],
            'bookstore': ['book', 'books', 'bookstore', 'literature'],
            'pharmacy': ['pharmacy', 'drugstore', 'health', 'medical'],
            'pet_shop': ['pet', 'animal', 'pet shop', 'pet store'],
            'hardware_store': ['hardware', 'tools', 'construction', 'building'],
            'furniture_store': ['furniture', 'home', 'interior', 'decor'],
            'gift_shop': ['gift', 'gifts', 'souvenir', 'present']
        }
        
        bio_lower = bio.lower()
        for business_type, keywords in business_types.items():
            if any(keyword in bio_lower for keyword in keywords):
                return business_type
        
        return 'general_business'

    def format_pain_point(self, category: str, business_type: str) -> str:
        """Format pain point based on category and business type"""
        pain_point_formats = {
            'time_management': {
                'retail': 'manual inventory management taking too much time',
                'restaurant': 'phone orders and manual scheduling consuming too much time',
                'grocery': 'manual inventory tracking and order management',
                'salon': 'phone booking and manual scheduling taking too much time',
                'boutique': 'manual inventory and order management',
                'coffee_shop': 'manual order taking and scheduling',
                'bakery': 'manual order tracking and inventory management',
                'clothing_store': 'manual inventory and sales tracking',
                'jewelry_store': 'manual inventory and customer records',
                'bookstore': 'manual inventory and order management',
                'pharmacy': 'manual prescription and inventory tracking',
                'pet_shop': 'manual inventory and appointment scheduling',
                'hardware_store': 'manual inventory and order management',
                'furniture_store': 'manual inventory and delivery scheduling',
                'gift_shop': 'manual inventory and order tracking'
            },
            'customer_reach': {
                'retail': 'limited to local walk-in customers only',
                'restaurant': 'relying only on walk-in and phone customers',
                'grocery': 'limited to local customer base',
                'salon': 'relying on walk-in and phone bookings',
                'boutique': 'limited to local store visitors',
                'coffee_shop': 'relying on walk-in customers only',
                'bakery': 'limited to local customer reach',
                'clothing_store': 'restricted to local store traffic',
                'jewelry_store': 'limited to local customer base',
                'bookstore': 'restricted to local walk-in customers',
                'pharmacy': 'limited to local prescription customers',
                'pet_shop': 'restricted to local pet owners',
                'hardware_store': 'limited to local DIY customers',
                'furniture_store': 'restricted to local showroom visitors',
                'gift_shop': 'limited to local gift shoppers'
            },
            'online_presence': {
                'retail': 'no online store for 24/7 sales',
                'restaurant': 'no online ordering system',
                'grocery': 'no online shopping option',
                'salon': 'no online booking system',
                'boutique': 'no online shopping experience',
                'coffee_shop': 'no online ordering platform',
                'bakery': 'no online order system',
                'clothing_store': 'no online catalog',
                'jewelry_store': 'no online showcase',
                'bookstore': 'no online book catalog',
                'pharmacy': 'no online prescription system',
                'pet_shop': 'no online pet supplies store',
                'hardware_store': 'no online tool catalog',
                'furniture_store': 'no online furniture showcase',
                'gift_shop': 'no online gift store'
            }
        }
        
        # Get business-specific format or use general format
        formats = pain_point_formats.get(category, {})
        return formats.get(business_type, f"challenges with {category.replace('_', ' ')}")

    def get_business_specific_pain_points(self, business_type: str, bio: str) -> List[str]:
        """Get additional business-specific pain points based on context"""
        specific_pain_points = {
            'retail': [
                'limited store hours affecting sales',
                'manual inventory causing stock issues',
                'cash-only transactions limiting sales'
            ],
            'restaurant': [
                'phone orders causing errors',
                'manual reservation system leading to double bookings',
                'limited delivery options affecting revenue'
            ],
            'grocery': [
                'manual inventory tracking causing stockouts',
                'limited delivery options affecting customer convenience',
                'cash-only transactions limiting payment options'
            ],
            'salon': [
                'phone booking causing scheduling conflicts',
                'manual client records leading to errors',
                'limited marketing reach affecting new customer acquisition'
            ],
            'boutique': [
                'limited store hours affecting customer access',
                'manual inventory management causing stock issues',
                'no online shopping limiting sales potential'
            ],
            'coffee_shop': [
                'manual order taking causing errors',
                'limited delivery options affecting revenue',
                'manual loyalty program limiting customer retention'
            ],
            'bakery': [
                'manual order tracking causing errors',
                'limited delivery options affecting customer reach',
                'cash-only transactions limiting payment options'
            ],
            'clothing_store': [
                'no online shopping limiting sales',
                'manual inventory management causing stock issues',
                'limited store hours affecting customer access'
            ],
            'jewelry_store': [
                'no online catalog limiting sales',
                'manual inventory tracking causing errors',
                'limited store hours affecting customer access'
            ],
            'bookstore': [
                'no online shopping limiting sales',
                'manual inventory management causing stock issues',
                'limited store hours affecting customer access'
            ],
            'pharmacy': [
                'no online ordering affecting convenience',
                'manual prescription tracking causing errors',
                'limited delivery options affecting customer service'
            ],
            'pet_shop': [
                'no online shopping limiting sales',
                'manual inventory management causing stock issues',
                'limited store hours affecting customer access'
            ],
            'hardware_store': [
                'no online catalog limiting sales',
                'manual inventory management causing stock issues',
                'limited store hours affecting customer access'
            ],
            'furniture_store': [
                'no online catalog limiting sales',
                'manual inventory management causing stock issues',
                'limited store hours affecting customer access'
            ],
            'gift_shop': [
                'no online shopping limiting sales',
                'manual inventory management causing stock issues',
                'limited store hours affecting customer access'
            ]
        }
        
        # Get business-specific pain points
        points = specific_pain_points.get(business_type, [])
        
        # Filter based on bio context
        relevant_points = []
        for point in points:
            if any(word in bio.lower() for word in point.split()):
                relevant_points.append(point)
        
        return relevant_points[:2]  # Return top 2 most relevant points

    def get_default_pain_points(self) -> List[str]:
        """Get default pain points for general businesses"""
        return [
            'limited online presence affecting growth',
            'manual business operations taking too much time',
            'restricted to local customer base only'
        ]

    def generate_lead_outreach(self, lead_data: Dict) -> str:
        """Generate personalized outreach message for leads"""
        try:
            # Analyze user's specific pain points
            pain_points = self.analyze_user_pain_points(lead_data['username'])
            
            # Select appropriate template based on business type
            business_type = lead_data['business_type'].lower()
            template = self.select_dm_template(business_type, pain_points)
            
            prompt = f"""
            Generate a personalized DM for a business owner:
            
            Lead Info:
            - Username: {lead_data['username']}
            - Business Type: {business_type}
            - Identified Pain Points: {pain_points}
            
            Template Style: {template}
            
            Requirements:
            - Keep it under 280 characters
            - Be personal and relevant to their business type
            - Address their specific pain points: {pain_points}
            - Offer genuine value
            - Don't be pushy or salesy
            - Include a soft call-to-action
            - Use their business type in the message
            - Show understanding of their industry
            - Focus on solving their specific challenges
            """
            
            response = self.gemini_model.generate_content(prompt)
            message = response.text.strip()
            
            # Save DM to database
            self.save_dm(lead_data['id'], 'initial', message)
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error generating lead outreach: {e}")
            return None
    
    def select_dm_template(self, business_type: str, pain_points: List[str]) -> str:
        """Select appropriate DM template based on business type"""
        templates = {
            'retail': """
            Hi {username}! 👋 I noticed your {business_type} and thought you might be interested in how other local retailers are growing their business online. Many stores like yours are increasing sales by 40% with a professional website. Would you like to hear how?
            """,
            'restaurant': """
            Hi {username}! 👋 I help local restaurants like yours set up online ordering and delivery systems. Many of our restaurant clients have doubled their orders since going digital. Would you like to know how?
            """,
            'grocery': """
            Hi {username}! 👋 I help grocery stores modernize their business with online ordering and delivery systems. Many of our grocery clients have seen a 50% increase in customer reach. Would you like to learn more?
            """,
            'salon': """
            Hi {username}! 👋 I help salons and spas set up online booking systems. Our salon clients have reduced no-shows by 30% and increased bookings by 40%. Would you like to know how?
            """,
            'boutique': """
            Hi {username}! 👋 I help boutiques create beautiful online stores. Many of our boutique clients have expanded their customer base beyond their local area. Would you like to hear their success stories?
            """,
            'coffee_shop': """
            Hi {username}! 👋 I help coffee shops set up online ordering and loyalty programs. Many of our cafe clients have increased their daily orders by 35%. Would you like to know how?
            """,
            'bakery': """
            Hi {username}! 👋 I help bakeries set up online ordering and delivery systems. Many of our bakery clients have expanded their reach and increased orders by 45%. Would you like to learn more?
            """,
            'clothing_store': """
            Hi {username}! 👋 I help clothing stores create online catalogs and shopping experiences. Many of our retail clients have increased their sales by 50% with an online presence. Would you like to know how?
            """,
            'jewelry_store': """
            Hi {username}! 👋 I help jewelry stores showcase their products online. Many of our jewelry clients have expanded their customer base and increased sales by 40%. Would you like to hear how?
            """,
            'bookstore': """
            Hi {username}! 👋 I help bookstores create online catalogs and shopping experiences. Many of our bookstore clients have increased their reach and sales by 35%. Would you like to learn more?
            """,
            'pharmacy': """
            Hi {username}! 👋 I help pharmacies set up online ordering and prescription management systems. Many of our pharmacy clients have improved customer service and increased orders by 40%. Would you like to know how?
            """,
            'pet_shop': """
            Hi {username}! 👋 I help pet shops create online stores and delivery systems. Many of our pet shop clients have expanded their reach and increased sales by 45%. Would you like to hear how?
            """,
            'hardware_store': """
            Hi {username}! 👋 I help hardware stores create online catalogs and shopping experiences. Many of our hardware store clients have increased their sales by 40% with an online presence. Would you like to learn more?
            """,
            'furniture_store': """
            Hi {username}! 👋 I help furniture stores showcase their products online. Many of our furniture store clients have expanded their reach and increased sales by 50%. Would you like to know how?
            """,
            'gift_shop': """
            Hi {username}! 👋 I help gift shops create online stores and shopping experiences. Many of our gift shop clients have increased their sales by 45% with an online presence. Would you like to hear how?
            """
        }
        
        # Find matching template
        for key, template in templates.items():
            if key in business_type:
                return template
        
        # Default template if no match
        return """
        Hi {username}! 👋 I help local businesses like yours establish a strong online presence. Many of our clients have increased their customer base and sales by 40% with a professional website. Would you like to hear how?
        """
    
    def save_dm(self, lead_id: int, message_type: str, content: str):
        """Save DM to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO direct_messages (lead_id, message_type, content)
            VALUES (?, ?, ?)
        ''', (lead_id, message_type, content))
        
        # Update lead's DM status
        cursor.execute('''
            UPDATE leads 
            SET dm_status = 'sent', 
                dm_sent_at = CURRENT_TIMESTAMP,
                status = 'contacted'
            WHERE id = ?
        ''', (lead_id,))
        
        self.conn.commit()
    
    def send_follow_up_dm(self, lead_id: int):
        """Send follow-up DM to leads who haven't responded"""
        try:
            cursor = self.conn.cursor()
            
            # Get lead info
            cursor.execute('SELECT * FROM leads WHERE id = ?', (lead_id,))
            lead = cursor.fetchone()
            
            if not lead:
                return
            
            # Check if we should send follow-up
            if lead[13] >= 3:  # follow_up_count
                return  # Max follow-ups reached
            
            # Get time since last DM
            cursor.execute('''
                SELECT dm_sent_at FROM leads WHERE id = ?
            ''', (lead_id,))
            last_dm = cursor.fetchone()[0]
            
            if last_dm:
                last_dm_time = datetime.strptime(last_dm, '%Y-%m-%d %H:%M:%S')
                if datetime.now() - last_dm_time < timedelta(days=3):
                    return  # Too soon for follow-up
            
            # Generate follow-up message
            follow_up = self.generate_follow_up_message(lead)
            
            if follow_up:
                # Save follow-up DM
                self.save_dm(lead_id, 'follow_up', follow_up)
                
                # Update follow-up count
                cursor.execute('''
                    UPDATE leads 
                    SET follow_up_count = follow_up_count + 1,
                        last_follow_up = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (lead_id,))
                
                self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error sending follow-up DM: {e}")
    
    def generate_follow_up_message(self, lead) -> str:
        """Generate follow-up message based on lead type and previous interactions"""
        try:
            business_type = lead[4].lower()  # business_type column
            
            follow_up_templates = {
                'retail': "Hi {username}! 👋 Just following up about helping your {business_type} grow online. Many local retailers are seeing great results with our website solutions. Would you like to see some examples?",
                'restaurant': "Hi {username}! 👋 Following up about online ordering for your {business_type}. Many restaurants are increasing orders by 40% with our system. Would you like to know more?",
                'grocery': "Hi {username}! 👋 Just checking in about online ordering for your {business_type}. Many grocery stores are expanding their reach with our solutions. Would you like to see how?",
                'salon': "Hi {username}! 👋 Following up about online booking for your {business_type}. Many salons are reducing no-shows and increasing bookings with our system. Would you like to learn more?",
                'boutique': "Hi {username}! 👋 Just checking in about creating an online store for your {business_type}. Many boutiques are reaching new customers with our solutions. Would you like to see some examples?",
                'coffee_shop': "Hi {username}! 👋 Following up about online ordering for your {business_type}. Many cafes are increasing orders with our system. Would you like to know more?",
                'bakery': "Hi {username}! 👋 Just checking in about online ordering for your {business_type}. Many bakeries are expanding their reach with our solutions. Would you like to see how?",
                'clothing_store': "Hi {username}! 👋 Following up about creating an online catalog for your {business_type}. Many clothing stores are increasing sales with our system. Would you like to learn more?",
                'jewelry_store': "Hi {username}! 👋 Just checking in about showcasing your {business_type} online. Many jewelry stores are reaching new customers with our solutions. Would you like to see some examples?",
                'bookstore': "Hi {username}! 👋 Following up about creating an online catalog for your {business_type}. Many bookstores are increasing sales with our system. Would you like to know more?",
                'pharmacy': "Hi {username}! 👋 Just checking in about online ordering for your {business_type}. Many pharmacies are improving service with our solutions. Would you like to see how?",
                'pet_shop': "Hi {username}! 👋 Following up about creating an online store for your {business_type}. Many pet shops are expanding their reach with our system. Would you like to learn more?",
                'hardware_store': "Hi {username}! 👋 Just checking in about creating an online catalog for your {business_type}. Many hardware stores are increasing sales with our solutions. Would you like to see some examples?",
                'furniture_store': "Hi {username}! 👋 Following up about showcasing your {business_type} online. Many furniture stores are reaching new customers with our system. Would you like to know more?",
                'gift_shop': "Hi {username}! 👋 Just checking in about creating an online store for your {business_type}. Many gift shops are increasing sales with our solutions. Would you like to see how?"
            }
            
            # Find matching template
            template = None
            for key, t in follow_up_templates.items():
                if key in business_type:
                    template = t
                    break
            
            if not template:
                template = "Hi {username}! 👋 Just following up about helping your business grow online. Many local businesses are seeing great results with our website solutions. Would you like to see some examples?"
            
            # Format template with lead info
            message = template.format(
                username=lead[1],  # username column
                business_type=business_type
            )
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error generating follow-up message: {e}")
            return None
    
    def initiate_lead_outreach(self, lead):
        """Initiate outreach to high-value leads"""
        try:
            # Generate personalized message
            lead_data = {
                'id': lead[0],
                'username': lead[1],
                'platform': lead[2],
                'business_type': lead[4]
            }
            
            message = self.generate_lead_outreach(lead_data)
            
            if message:
                # Send DM using Twitter API
                try:
                    self.twitter_client.send_direct_message(
                        recipient_id=lead[1],
                        text=message
                    )
                    self.logger.info(f"Sent DM to {lead[1]}: {message[:50]}...")
                    
                    # Schedule follow-up
                    schedule.every(3).days.do(self.send_follow_up_dm, lead[0])
                    
                except Exception as e:
                    self.logger.error(f"Error sending DM: {e}")
        
        except Exception as e:
            self.logger.error(f"Error initiating lead outreach: {e}")
    
    def send_daily_report(self):
        """Send daily performance report via email"""
        try:
            # Gather analytics
            analytics = self.get_daily_analytics()
            
            # Generate report
            report = self.generate_daily_report(analytics)
            
            # Send email
            self.send_email_report(report)
            
        except Exception as e:
            self.logger.error(f"Error sending daily report: {e}")
    
    def get_daily_analytics(self) -> Dict:
        """Get daily analytics from database"""
        cursor = self.conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Posts today
        cursor.execute('''
            SELECT COUNT(*) FROM posts 
            WHERE DATE(created_at) = ?
        ''', (today,))
        posts_today = cursor.fetchone()[0]
        
        # Total engagement
        cursor.execute('''
            SELECT SUM(likes + retweets + mentions) FROM posts 
            WHERE DATE(created_at) = ?
        ''', (today,))
        total_engagement = cursor.fetchone()[0] or 0
        
        # New leads
        cursor.execute('''
            SELECT COUNT(*) FROM leads 
            WHERE DATE(last_interaction) = ?
        ''', (today,))
        new_leads = cursor.fetchone()[0]
        
        # Trending topics discovered
        cursor.execute('''
            SELECT COUNT(*) FROM trends 
            WHERE DATE(discovered_at) = ?
        ''', (today,))
        trends_discovered = cursor.fetchone()[0]
        
        return {
            'posts_today': posts_today,
            'total_engagement': total_engagement,
            'new_leads': new_leads,
            'trends_discovered': trends_discovered,
            'date': today
        }
    
    def generate_daily_report(self, analytics: Dict) -> str:
        """Generate HTML daily report"""
        report = f"""
        <html>
        <body>
        <h2>🤖 JustEmma AI Agent Daily Report - {analytics['date']}</h2>
        
        <h3>📊 Performance Metrics</h3>
        <ul>
            <li><strong>Posts Created:</strong> {analytics['posts_today']}</li>
            <li><strong>Total Engagement:</strong> {analytics['total_engagement']}</li>
            <li><strong>New Leads:</strong> {analytics['new_leads']}</li>
            <li><strong>Trends Discovered:</strong> {analytics['trends_discovered']}</li>
        </ul>
        
        <h3>🎯 Today's Achievements</h3>
        <ul>
            <li>✅ Automated social media posting</li>
            <li>✅ Community engagement and networking</li>
            <li>✅ Lead generation and analysis</li>
            <li>✅ Trend monitoring and content optimization</li>
        </ul>
        
        <h3>🚀 Next Steps</h3>
        <ul>
            <li>Continue engaging with high-value prospects</li>
            <li>Optimize content based on engagement patterns</li>
            <li>Follow up on potential leads</li>
            <li>Monitor emerging trends for content opportunities</li>
        </ul>
        
        <p><em>Report generated automatically by JustEmma AI Agent</em></p>
        </body>
        </html>
        """
        
        return report
    
    def schedule_posts(self):
        """Schedule posts throughout the day"""
        post_types = [
            (PostType.GREETING, "09:00"),
            (PostType.BUSINESS_TIP, "11:00"),
            (PostType.SUCCESS_STORY, "13:00"),
            (PostType.ECOMMERCE_TIP, "15:00"),
            (PostType.DIGITAL_MARKETING, "17:00"),
            (PostType.BUSINESS_GROWTH, "19:00"),
            (PostType.CUSTOMER_ENGAGEMENT, "21:00"),
            (PostType.LOCAL_BUSINESS, "23:00")
        ]
        
        for post_type, time_str in post_types:
            schedule.every().day.at(time_str).do(self.create_and_post, post_type)
        
        # Schedule other activities
        schedule.every(2).hours.do(self.scrape_trends)
        schedule.every(3).hours.do(self.engage_with_community)
        schedule.every().day.at("23:55").do(self.send_daily_report)
    
    def create_and_post(self, post_type: PostType):
        """Create and post content"""
        try:
            # Generate content
            content = self.generate_content(post_type)
            
            # Post to Twitter
            post_id = self.post_to_twitter(content, post_type)
            
            if post_id:
                self.logger.info(f"Successfully posted {post_type.value}: {content[:50]}...")
                
                # Mark first run as complete
                if self.first_run and post_type == PostType.GREETING:
                    self.first_run = False
                    self.logger.info("🎉 First run completed successfully!")
            else:
                self.logger.warning(f"Failed to post {post_type.value}")
                
        except Exception as e:
            self.logger.error(f"Error creating and posting content: {e}")
    
    def monitor_mentions_and_dms(self):
        """Monitor mentions and DMs for engagement opportunities"""
        try:
            # Get mentions
            mentions = self.twitter_client.get_mentions(max_results=20)
            
            if mentions.data:
                for mention in mentions.data:
                    self.handle_mention(mention)
                    time.sleep(1)  # Rate limiting
            
        except Exception as e:
            self.logger.error(f"Error monitoring mentions: {e}")
    
    def handle_mention(self, mention):
        """Handle mentions and replies"""
        try:
            # Generate contextual response
            response = self.generate_contextual_reply(mention.text)
            
            if response:
                # Reply to mention
                self.twitter_client.create_tweet(
                    text=response,
                    in_reply_to_tweet_id=mention.id
                )
                
                self.logger.info(f"Replied to mention from {mention.author_id}")
        
        except Exception as e:
            self.logger.error(f"Error handling mention: {e}")
    
    def smart_follow_strategy(self):
        """Implement smart following strategy"""
        try:
            # Search for developers and tech professionals
            search_queries = [
                "web developer",
                "frontend developer", 
                "fullstack developer",
                "AI engineer",
                "startup founder",
                "tech entrepreneur"
            ]
            
            follows_today = 0
            max_follows = 50
            
            for query in search_queries:
                if follows_today >= max_follows:
                    break
                    
                users = self.twitter_client.search_users(query=query, max_results=10)
                
                if users.data:
                    for user in users.data:
                        if follows_today >= max_follows:
                            break
                            
                        if self.should_follow_user(user):
                            self.follow_user(user)
                            follows_today += 1
                            time.sleep(30)  # Rate limiting
        
        except Exception as e:
            self.logger.error(f"Error in smart follow strategy: {e}")
    
    def should_follow_user(self, user) -> bool:
        """Determine if we should follow a user"""
        try:
            bio = user.description or ""
            
            # Positive indicators
            positive_keywords = [
                'developer', 'programmer', 'engineer', 'founder', 'startup',
                'entrepreneur', 'freelancer', 'consultant', 'tech', 'web'
            ]
            
            # Negative indicators
            negative_keywords = [
                'bot', 'spam', 'marketing', 'adult', 'crypto pump'
            ]
            
            bio_lower = bio.lower()
            
            # Check for negative indicators first
            for keyword in negative_keywords:
                if keyword in bio_lower:
                    return False
            
            # Check for positive indicators
            for keyword in positive_keywords:
                if keyword in bio_lower:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating user for following: {e}")
            return False
    
    def follow_user(self, user):
        """Follow a user and save as potential lead"""
        try:
            self.twitter_client.follow_user(user.id)
            
            # Save as potential lead
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO leads (username, platform, profile_url, business_type, contact_info, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user.username,
                'twitter',
                f"https://twitter.com/{user.username}",
                user.description,
                user.description,
                'followed'
            ))
            
            self.conn.commit()
            self.logger.info(f"Followed user: {user.username}")
            
        except Exception as e:
            self.logger.error(f"Error following user: {e}")
    
    def analyze_competitors(self):
        """Analyze competitors and their strategies"""
        try:
            competitors = [
                "buildspace", "devto", "freecodecamp", "reactjs", "vuejs"
            ]
            
            for competitor in competitors:
                self.analyze_competitor_account(competitor)
                time.sleep(60)  # Rate limiting
        
        except Exception as e:
            self.logger.error(f"Error analyzing competitors: {e}")
    
    def analyze_competitor_account(self, username: str):
        """Analyze a specific competitor account"""
        try:
            # Get user info
            user = self.twitter_client.get_user(username=username, user_fields=['public_metrics'])
            
            if user.data:
                # Get recent tweets
                tweets = self.twitter_client.get_users_tweets(
                    user.data.id,
                    max_results=10,
                    tweet_fields=['public_metrics', 'created_at']
                )
                
                if tweets.data:
                    # Analyze engagement patterns
                    total_engagement = 0
                    best_performing_tweet = None
                    max_engagement = 0
                    
                    for tweet in tweets.data:
                        engagement = (
                            tweet.public_metrics['like_count'] +
                            tweet.public_metrics['retweet_count'] +
                            tweet.public_metrics['reply_count']
                        )
                        
                        total_engagement += engagement
                        
                        if engagement > max_engagement:
                            max_engagement = engagement
                            best_performing_tweet = tweet
                    
                    avg_engagement = total_engagement / len(tweets.data)
                    
                    # Learn from best performing content
                    if best_performing_tweet:
                        self.learn_from_successful_content(best_performing_tweet.text)
                    
                    self.logger.info(f"Analyzed {username}: Avg engagement {avg_engagement}")
        
        except Exception as e:
            self.logger.error(f"Error analyzing competitor {username}: {e}")
    
    def learn_from_successful_content(self, content: str):
        """Learn patterns from successful content"""
        try:
            # Extract hashtags
            hashtags = re.findall(r'#\w+', content)
            
            # Extract content themes
            words = content.lower().split()
            tech_words = [w for w in words if w in [
                'javascript', 'python', 'react', 'vue', 'nodejs', 'ai', 'ml',
                'webdev', 'coding', 'programming', 'developer', 'tech'
            ]]
            
            # Save insights to database
            cursor = self.conn.cursor()
            for hashtag in hashtags:
                cursor.execute('''
                    INSERT OR REPLACE INTO trends (trend_topic, trend_score, platform)
                    VALUES (?, ?, ?)
                ''', (hashtag[1:], 9, 'competitor_analysis'))
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error learning from content: {e}")
    
    def optimize_posting_times(self):
        """Analyze and optimize posting times based on engagement"""
        try:
            cursor = self.conn.cursor()
            
            # Get historical engagement data
            cursor.execute('''
                SELECT 
                    strftime('%H', created_at) as hour,
                    AVG(likes + retweets + mentions) as avg_engagement
                FROM posts 
                WHERE created_at >= date('now', '-30 days')
                GROUP BY hour
                ORDER BY avg_engagement DESC
            ''')
            
            results = cursor.fetchall()
            
            if results:
                best_hours = [int(row[0]) for row in results[:5]]
                self.logger.info(f"Best posting hours: {best_hours}")
                
                # Update posting schedule dynamically
                self.update_posting_schedule(best_hours)
        
        except Exception as e:
            self.logger.error(f"Error optimizing posting times: {e}")
    
    def update_posting_schedule(self, best_hours: List[int]):
        """Update posting schedule based on optimization"""
        try:
            # Clear existing schedule
            schedule.clear()
            
            # Reschedule with optimized times
            post_types = [
                PostType.GREETING, PostType.BUSINESS_TIP, PostType.SUCCESS_STORY,
                PostType.ECOMMERCE_TIP, PostType.DIGITAL_MARKETING, PostType.BUSINESS_GROWTH,
                PostType.CUSTOMER_ENGAGEMENT, PostType.LOCAL_BUSINESS
            ]
            
            for i, post_type in enumerate(post_types):
                if i < len(best_hours):
                    hour = best_hours[i]
                    schedule.every().day.at(f"{hour:02d}:00").do(self.create_and_post, post_type)
            
            # Reschedule other activities
            schedule.every(2).hours.do(self.scrape_trends)
            schedule.every(3).hours.do(self.engage_with_community)
            schedule.every().day.at("23:55").do(self.send_daily_report)
            schedule.every(4).hours.do(self.monitor_mentions_and_dms)
            schedule.every().day.at("08:00").do(self.smart_follow_strategy)
            schedule.every().day.at("20:00").do(self.analyze_competitors)
            
            self.logger.info("Posting schedule optimized!")
            
        except Exception as e:
            self.logger.error(f"Error updating posting schedule: {e}")
    
    def create_visual_content(self, text: str) -> Optional[bytes]:
        """Create visual content for posts"""
        try:
            # Create a simple code-themed image
            img = Image.new('RGB', (800, 400), color='#1a1a1a')
            draw = ImageDraw.Draw(img)
            
            # Try to load a font (fallback to default if not available)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Add title
            draw.text((50, 50), "JustEmma", fill='#00ff88', font=font)
            draw.text((50, 80), "Web Development & AI", fill='#ffffff', font=small_font)
            
            # Add main text (wrapped)
            y_position = 150
            words = text.split()
            line = ""
            
            for word in words:
                if len(line + word) < 40:  # Wrap at 40 characters
                    line += word + " "
                else:
                    draw.text((50, y_position), line, fill='#cccccc', font=small_font)
                    y_position += 25
                    line = word + " "
                    
                    if y_position > 320:  # Don't overflow image
                        break
            
            # Add final line
            if line:
                draw.text((50, y_position), line, fill='#cccccc', font=small_font)
            
            # Add decorative elements
            draw.rectangle([40, 40, 760, 44], fill='#00ff88')
            draw.rectangle([40, 356, 760, 360], fill='#00ff88')
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return img_byte_arr
            
        except Exception as e:
            self.logger.error(f"Error creating visual content: {e}")
            return None
    
    def advanced_lead_scoring(self):
        """Advanced lead scoring and prioritization"""
        try:
            cursor = self.conn.cursor()
            
            # Get all leads
            cursor.execute('SELECT * FROM leads WHERE status != "contacted"')
            leads = cursor.fetchall()
            
            for lead in leads:
                score = self.calculate_lead_score(lead)
                
                # Update lead score in database
                cursor.execute('''
                    UPDATE leads SET engagement_level = ? WHERE id = ?
                ''', (score, lead[0]))
            
            self.conn.commit()
            
            # Get top leads for outreach
            cursor.execute('''
                SELECT * FROM leads 
                WHERE engagement_level > 7 AND status = 'new'
                ORDER BY engagement_level DESC
                LIMIT 5
            ''')
            
            top_leads = cursor.fetchall()
            
            for lead in top_leads:
                self.initiate_lead_outreach(lead)
                time.sleep(300)  # 5 minute delay between outreach
        
        except Exception as e:
            self.logger.error(f"Error in advanced lead scoring: {e}")
    
    def calculate_lead_score(self, lead) -> int:
        """Calculate lead score based on business indicators"""
        score = 0
        bio = lead[4] or ""  # business_type column
        
        # Business type indicators (higher score)
        business_keywords = {
            'retail': 8, 'store': 7, 'shop': 7, 'boutique': 8,
            'restaurant': 9, 'cafe': 8, 'coffee shop': 8,
            'grocery': 9, 'market': 8, 'supermarket': 9,
            'salon': 7, 'spa': 7, 'beauty': 7,
            'fitness': 7, 'gym': 7, 'wellness': 7,
            'bakery': 8, 'cafe': 8, 'food': 8,
            'clothing': 8, 'fashion': 8, 'apparel': 8,
            'jewelry': 9, 'accessories': 7,
            'bookstore': 7, 'books': 7,
            'pharmacy': 9, 'health': 8,
            'pet shop': 7, 'pet store': 7,
            'hardware': 8, 'tools': 7,
            'furniture': 8, 'home': 7,
            'gift shop': 7, 'gifts': 7
        }
        
        # Online presence indicators
        online_keywords = {
            'website': 10, 'online': 8, 'ecommerce': 10,
            'digital': 7, 'web': 8, 'internet': 7,
            'social media': 6, 'online store': 9,
            'e-shop': 9, 'e-store': 9
        }
        
        # Business size indicators
        size_keywords = {
            'small business': 8, 'local business': 7,
            'family owned': 8, 'independent': 7,
            'startup': 6, 'new business': 6
        }
        
        # Check for business type
        bio_lower = bio.lower()
        for keyword, points in business_keywords.items():
            if keyword in bio_lower:
                score += points
                break  # Only count highest matching business type
        
        # Check for online presence
        for keyword, points in online_keywords.items():
            if keyword in bio_lower:
                score += points
        
        # Check for business size
        for keyword, points in size_keywords.items():
            if keyword in bio_lower:
                score += points
        
        # Additional scoring factors
        if 'need website' in bio_lower or 'looking for website' in bio_lower:
            score += 5  # High priority if they're actively looking
        if 'help' in bio_lower or 'assistance' in bio_lower:
            score += 3  # Shows they're seeking solutions
        if len(bio) > 50:  # Detailed bio indicates active user
            score += 2
        
        return min(score, 10)  # Cap at 10
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring and optimization"""
        while self.is_running:
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Additional monitoring
                if datetime.now().minute % 30 == 0:  # Every 30 minutes
                    self.optimize_posting_times()
                
                if datetime.now().hour % 6 == 0 and datetime.now().minute == 0:  # Every 6 hours
                    self.advanced_lead_scoring()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def start_agent(self):
        """Start the AI agent"""
        try:
            self.logger.info("🚀 Starting JustEmma AI Social Media Agent...")
            
            # Initial setup
            self.scrape_trends()
            self.schedule_posts()
            
            # Post initial greeting if first run
            if self.first_run:
                greeting_content = self.generate_content(PostType.GREETING)
                self.post_to_twitter(greeting_content, PostType.GREETING)
                self.first_run = False
            
            self.is_running = True
            
            # Start monitoring thread
            monitoring_thread = threading.Thread(target=self.run_continuous_monitoring)
            monitoring_thread.daemon = True
            monitoring_thread.start()
            
            self.logger.info("✅ AI Agent started successfully!")
            self.logger.info("🎯 Monitoring social media, generating content, and building your network...")
            
            # Keep main thread alive
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down AI Agent...")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"Error starting agent: {e}")
    
    def stop_agent(self):
        """Stop the AI agent"""
        self.is_running = False
        self.conn.close()
        self.logger.info("AI Agent stopped successfully")

    def identify_pain_points(self, business_type: str) -> List[str]:
        """Identify common pain points for a specific business type"""
        pain_points = {
            'retail': [
                'manual inventory management taking too much time',
                'limited to local walk-in customers only',
                'no online store for 24/7 sales'
            ],
            'restaurant': [
                'phone orders causing errors',
                'manual reservation system leading to double bookings',
                'no online ordering system'
            ],
            'grocery': [
                'manual inventory tracking causing stockouts',
                'limited delivery options affecting customer convenience',
                'no online shopping option'
            ],
            'salon': [
                'phone booking causing scheduling conflicts',
                'manual client records leading to errors',
                'no online booking system'
            ],
            'coffee_shop': [
                'manual order taking causing errors',
                'limited delivery options affecting revenue',
                'no online ordering platform'
            ],
            'bakery': [
                'manual order tracking causing errors',
                'limited delivery options affecting customer reach',
                'no online order system'
            ],
            'clothing_store': [
                'no online shopping limiting sales',
                'manual inventory management causing stock issues',
                'no online catalog'
            ],
            'jewelry_store': [
                'no online catalog limiting sales',
                'manual inventory tracking causing errors',
                'no online showcase'
            ],
            'bookstore': [
                'no online shopping limiting sales',
                'manual inventory management causing stock issues',
                'no online book catalog'
            ],
            'pharmacy': [
                'no online ordering affecting convenience',
                'manual prescription tracking causing errors',
                'no online prescription system'
            ],
            'pet_shop': [
                'no online shopping limiting sales',
                'manual inventory management causing stock issues',
                'no online pet supplies store'
            ],
            'hardware_store': [
                'no online catalog limiting sales',
                'manual inventory management causing stock issues',
                'no online tool catalog'
            ],
            'furniture_store': [
                'no online catalog limiting sales',
                'manual inventory management causing stock issues',
                'no online furniture showcase'
            ],
            'gift_shop': [
                'no online shopping limiting sales',
                'manual inventory management causing stock issues',
                'no online gift store'
            ]
        }
        
        return pain_points.get(business_type, [
            'limited online presence affecting growth',
            'manual business operations taking too much time',
            'restricted to local customer base only'
        ])

def initialize_agent():
    """Initialize the agent in the background"""
    try:
        print("[INIT] Starting agent initialization...")
        # Initialize the agent
        agent = SocialMediaAgent()
        print("[INIT] Agent object created successfully")
        
        # Store agent in Flask app context
        app.agent = agent
        print("[INIT] Agent stored in Flask app context")
        
        # Update initialization status
        app.initialization_status['status'] = 'running'
        app.initialization_status['agent_ready'] = True
        print("[INIT] Initialization status updated")
        
        # Start the agent in a separate thread
        print("[INIT] Starting agent thread...")
        agent_thread = threading.Thread(target=agent.start_agent, name="AgentThread")
        agent_thread.daemon = False  # Make it non-daemon so it keeps running
        agent_thread.start()
        print("[INIT] Agent thread started")
        
        # Add a startup endpoint to check agent status
        @app.route('/agent-status')
        def agent_status():
            """Detailed agent status endpoint"""
            try:
                if not hasattr(app, 'agent'):
                    return jsonify({
                        'status': 'not_initialized',
                        'message': 'Agent not initialized yet',
                        'timestamp': datetime.now().isoformat()
                    }), 503
                
                thread_status = "unknown"
                if hasattr(agent_thread, 'is_alive'):
                    thread_status = "alive" if agent_thread.is_alive() else "dead"
                
                return jsonify({
                    'status': 'running' if app.agent.is_running else 'stopped',
                    'initialization_status': app.initialization_status['status'],
                    'agent_ready': app.initialization_status['agent_ready'],
                    'first_run': app.agent.first_run,
                    'thread_status': thread_status,
                    'thread_name': agent_thread.name,
                    'timestamp': datetime.now().isoformat(),
                    'uptime_seconds': (datetime.now() - datetime.fromisoformat(app.initialization_status['start_time'])).total_seconds()
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
    except Exception as e:
        error_msg = f"Error initializing agent: {str(e)}"
        print(f"[ERROR] {error_msg}")
        app.initialization_status['status'] = f'error: {str(e)}'
        app.initialization_status['agent_ready'] = False
        # Don't raise the exception, just log it
        return False

def start_application():
    """Start the application with proper initialization"""
    print("[STARTUP] Starting application...")
    
    # Start agent initialization in background
    print("[STARTUP] Starting initialization thread...")
    init_thread = threading.Thread(target=initialize_agent, name="InitThread")
    init_thread.daemon = False  # Make it non-daemon
    init_thread.start()
    print("[STARTUP] Initialization thread started")
    
    # Get port from environment variable
    port = int(os.getenv('PORT', 10000))
    print(f"[STARTUP] Starting server on port {port}")
    
    # Configure Gunicorn options
    options = {
        'bind': f'0.0.0.0:{port}',
        'workers': 1,  # Single worker since we're running the agent
        'timeout': 120,
        'accesslog': '-',
        'errorlog': '-',
        'loglevel': 'info',
        'preload_app': True,
        'reload': False,
        'worker_class': 'sync',
        'worker_connections': 1000,
        'backlog': 2048,
        'keepalive': 5,
        'max_requests': 1000,
        'max_requests_jitter': 50
    }
    
    print("[STARTUP] Starting Gunicorn server...")
    StandaloneApplication(app, options).run()

if __name__ == '__main__':
    start_application()
else:
    # When running with Gunicorn
    print("[GUNICORN] Starting application...")
    start_application()