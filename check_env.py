import os
import sys

required_vars = [
    'SENDGRID_API_KEY',
    'TWITTER_API_KEY',
    'TWITTER_API_SECRET',
    'TWITTER_BEARER_TOKEN',
    'TWITTER_ACCESS_TOKEN',
    'TWITTER_ACCESS_SECRET',
    'EMAIL_FROM',
    'EMAIL_TO',
    'REPORT_EMAIL'
]

# Only check for SendGrid if SENDGRID_API_KEY is present
if os.getenv('SENDGRID_API_KEY'):
    sendgrid_vars = ['EMAIL_FROM', 'EMAIL_TO', 'REPORT_EMAIL']
    missing_vars = [var for var in sendgrid_vars if not os.getenv(var)]
    if missing_vars:
        print("Warning: SendGrid email functionality will be disabled. Missing SendGrid variables:")
        for var in missing_vars:
            print(f"  - {var}")
else:
    print("Note: SendGrid email functionality is disabled (no SENDGRID_API_KEY)")

# Check core required variables (excluding email)
core_vars = [var for var in required_vars if var not in ['SENDGRID_API_KEY', 'EMAIL_FROM', 'EMAIL_TO', 'REPORT_EMAIL']]
missing_core = [var for var in core_vars if not os.getenv(var)]

if missing_core:
    print("Error: The following required environment variables are not set:")
    for var in missing_core:
        print(f"  - {var}")
    sys.exit(1)

print("All required environment variables are set.")
sys.exit(0) 