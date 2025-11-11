"""Data fetching script for nanoGPT training pipeline."""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import time

try:
    import requests
    import feedparser
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing required packages...")
    os.system("pip install requests feedparser beautifulsoup4")
    import requests
    import feedparser
    from bs4 import BeautifulSoup


class DataFetcher:
    def __init__(self, output_dir: str, days: int = 1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.days = days
        self.cutoff_date = datetime.now() - timedelta(days=days)
    
    def fetch_news(self, api_key: str = None) -> List[str]:
        texts = []
        
        # Use RSS feeds (no API key needed)
        rss_feeds = [
            "http://rss.cnn.com/rss/cnn_topstories.rss",
            "http://feeds.bbci.co.uk/news/rss.xml",
            "https://news.ycombinator.com/rss",
        ]
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title}\n\n{summary}".strip()
                    if text:
                        texts.append(text)
                print(f"✅ Fetched {len(feed.entries[:20])} items from {feed_url}")
                time.sleep(1)
            except Exception as e:
                print(f"⚠️  RSS feed error ({feed_url}): {e}")
        
        return texts
    
    def fetch_arxiv(self) -> List[str]:
        texts = []
        base_url = "http://export.arxiv.org/api/query"
        categories = ["cs.AI", "cs.LG"]
        
        for category in categories:
            try:
                params = {
                    "search_query": f"cat:{category}",
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": 30
                }
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
                
                for entry in feed.entries:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"Title: {title}\n\nAbstract: {summary}".strip()
                    if text:
                        texts.append(text)
                
                print(f"✅ Fetched {len(feed.entries)} papers from arXiv {category}")
                time.sleep(3)
            except Exception as e:
                print(f"⚠️  arXiv error ({category}): {e}")
        
        return texts
    
    def fetch_reddit(self, client_id: str = None, client_secret: str = None) -> List[str]:
        texts = []
        subreddits = ["technology", "science"]
        
        for subreddit in subreddits:
            try:
                feed_url = f"https://www.reddit.com/r/{subreddit}/.rss"
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:
                    title = entry.get("title", "")
                    summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()
                    text = f"{title}\n\n{summary}".strip()
                    if text and len(text) > 50:
                        texts.append(text)
                
                print(f"✅ Fetched {len(feed.entries[:20])} posts from r/{subreddit}")
                time.sleep(2)
            except Exception as e:
                print(f"⚠️  Reddit RSS error (r/{subreddit}): {e}")
        
        return texts
    
    def fetch_github(self) -> List[str]:
        texts = []
        topics = ["machine-learning", "artificial-intelligence"]
        base_url = "https://api.github.com/search/repositories"
        
        for topic in topics:
            try:
                params = {
                    "q": f"topic:{topic} pushed:>{self.cutoff_date.strftime('%Y-%m-%d')}",
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 15
                }
                headers = {"Accept": "application/vnd.github.v3+json"}
                response = requests.get(base_url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                
                repos = response.json().get("items", [])
                for repo in repos:
                    name = repo.get("full_name", "")
                    description = repo.get("description", "")
                    text = f"Repository: {name}\n\n{description}"
                    if text:
                        texts.append(text)
                
                print(f"✅ Fetched {len(repos)} repositories for topic '{topic}'")
                time.sleep(5)
            except Exception as e:
                print(f"⚠️  GitHub error ({topic}): {e}")
        
        return texts
    
    def save_texts(self, texts: List[str], source: str):
        if not texts:
            print(f"⚠️  No texts fetched from {source}")
            return
        
        output_file = self.output_dir / f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                text = text.strip()
                if text:
                    f.write(text + "\n\n" + "="*50 + "\n\n")
        
        print(f"💾 Saved {len(texts)} texts to {output_file}")
        
        metadata = {
            "source": source,
            "count": len(texts),
            "timestamp": datetime.now().isoformat(),
            "days": self.days
        }
        
        metadata_file = self.output_dir / f"{source}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True,
                        choices=["news", "arxiv", "reddit", "github"])
    parser.add_argument("--output-dir", type=str, default="./data/raw")
    parser.add_argument("--days", type=int, default=1)
    args = parser.parse_args()
    
    news_api_key = os.environ.get("NEWS_API_KEY")
    reddit_client_id = os.environ.get("REDDIT_CLIENT_ID")
    reddit_client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    
    fetcher = DataFetcher(args.output_dir, args.days)
    
    print(f"\n🚀 Fetching {args.source} data from last {args.days} day(s)...\n")
    
    if args.source == "news":
        texts = fetcher.fetch_news(news_api_key)
    elif args.source == "arxiv":
        texts = fetcher.fetch_arxiv()
    elif args.source == "reddit":
        texts = fetcher.fetch_reddit(reddit_client_id, reddit_client_secret)
    elif args.source == "github":
        texts = fetcher.fetch_github()
    else:
        print(f"❌ Unknown source: {args.source}")
        sys.exit(1)
    
    if texts:
        fetcher.save_texts(texts, args.source)
        print(f"\n✅ Successfully fetched and saved {len(texts)} texts")
    else:
        print(f"\n❌ Failed to fetch data from {args.source}")
        sys.exit(1)


if __name__ == "__main__":
    main()
