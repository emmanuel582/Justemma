# 🤖 JustEmma - AI Social Media Agent

JustEmma is an advanced AI-powered social media agent designed to help businesses grow their online presence through automated content generation, lead generation, and community engagement.

## 🌟 Features

- **Autonomous Posting**: 8 posts per day with business-focused content
- **Smart Community Engagement**: Intelligent interaction with relevant users
- **Lead Generation**: Automated identification and scoring of potential business leads
- **Trend Monitoring**: Real-time analysis of industry trends
- **Competitor Analysis**: Tracking and learning from competitor strategies
- **Performance Optimization**: Dynamic adjustment of posting times and content
- **Daily Reports**: Automated performance analytics and reporting
- **Visual Content Creation**: AI-generated images for social media posts

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/emmanuel582/JustEmma.git
cd JustEmma
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```env
# Twitter API
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_BEARER_TOKEN=your_bearer_token
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret
TWITTER_CLIENT_ID=your_client_id
TWITTER_CLIENT_SECRET=your_client_secret

# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent

# AI APIs
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token

# Email
SENDGRID_API_KEY=your_sendgrid_api_key
EMAIL_USER=your_email
REPORT_EMAIL=your_report_email
```

5. Run the agent:
```bash
python justemma.py
```

## 📋 Requirements

- Python 3.10 or higher
- SQLite3
- Internet connection
- Valid API keys for:
  - Twitter
  - Reddit
  - Gemini
  - OpenAI
  - HuggingFace
  - SendGrid

## 🛠️ Configuration

The agent can be configured through environment variables or by modifying the `Config` class in `justemma.py`. Key configuration options include:

- `DAILY_POSTS`: Number of posts per day (default: 8)
- `ENGAGEMENT_LIMIT`: Maximum daily engagements (default: 200)
- `FOLLOW_LIMIT`: Maximum daily follows (default: 50)

## 📊 Database Structure

The agent uses SQLite with the following tables:
- `posts`: Stores all social media posts
- `leads`: Tracks potential business leads
- `direct_messages`: Records DM conversations
- `analytics`: Stores performance metrics
- `trends`: Tracks trending topics

## 🔒 Security

- API keys are stored in environment variables
- Database is local and encrypted
- Rate limiting implemented for all API calls
- Secure error handling and logging

## 📈 Performance

- Automated content generation using AI
- Smart scheduling based on engagement patterns
- Lead scoring and prioritization
- Trend analysis and adaptation
- Performance monitoring and optimization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Google for Gemini AI
- HuggingFace for sentiment analysis
- Twitter and Reddit APIs
- All contributors and users

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 🔄 Updates

The agent is regularly updated with:
- New AI models
- Improved content generation
- Enhanced lead scoring
- Better trend analysis
- Security patches
- Performance optimizations 