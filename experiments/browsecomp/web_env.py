"""Deterministic web environment with on-disk caching for BrowseComp.

Provides web_search and web_fetch tools that cache all results for
reproducibility and determinism.
"""

import hashlib
import json
import os
import re
import urllib.parse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import requests
from bs4 import BeautifulSoup


# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "browsecomp"


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        return cls(**data)


@dataclass
class SearchResponse:
    """Response from a web search."""
    query: str
    results: List[SearchResult]
    cached: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "cached": self.cached,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResponse":
        return cls(
            query=data["query"],
            results=[SearchResult.from_dict(r) for r in data["results"]],
            cached=data.get("cached", True),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class FetchResult:
    """Result from fetching a URL."""
    url: str
    final_url: str  # After redirects
    status_code: int
    title: str
    text_content: str  # Extracted text (not raw HTML)
    cached: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FetchResult":
        return cls(**data)


def _normalize_query(query: str) -> str:
    """Normalize a search query for cache key generation."""
    # Lowercase, strip whitespace, collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', query.lower().strip())
    return normalized


def _normalize_url(url: str) -> str:
    """Normalize a URL for cache key generation."""
    parsed = urllib.parse.urlparse(url)
    # Remove trailing slashes, lowercase host
    normalized = urllib.parse.urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip('/') or '/',
        parsed.params,
        parsed.query,
        '',  # Remove fragment
    ))
    return normalized


def _hash_key(key: str) -> str:
    """Create a hash from a cache key."""
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]


def _extract_text_from_html(html: str, max_chars: int = 50000) -> str:
    """Extract readable text from HTML content."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        element.decompose()
    
    # Get text and clean it up
    text = soup.get_text(separator='\n', strip=True)
    
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Truncate if too long
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... content truncated ...]"
    
    return text


def _extract_title_from_html(html: str) -> str:
    """Extract page title from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text(strip=True)
    return ""


class WebEnvironment:
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        search_api_key: Optional[str] = None,
        search_engine: str = "serper",  # Options: "serper", "google_custom"
        user_agent: str = "Mozilla/5.0 (compatible; BrowseCompBot/1.0)",
        request_timeout: int = 30,
    ):
        """Initialize web environment.
        
        Args:
            cache_dir: Directory for caching results. Defaults to .cache/browsecomp/
            search_api_key: API key for search service.
            search_engine: Search engine to use ("serper" or "google_custom").
            user_agent: User agent string for HTTP requests.
            request_timeout: Timeout for HTTP requests in seconds.
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.search_api_key = search_api_key or os.getenv("SERPER_API_KEY")
        self.search_engine = search_engine
        self.user_agent = user_agent
        self.request_timeout = request_timeout
        
        # Create cache directories
        self.search_cache_dir = self.cache_dir / "search"
        self.fetch_cache_dir = self.cache_dir / "fetch"
        self.search_cache_dir.mkdir(parents=True, exist_ok=True)
        self.fetch_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_search_cache_path(self, query: str) -> Path:
        """Get cache file path for a search query."""
        normalized = _normalize_query(query)
        hash_key = _hash_key(normalized)
        return self.search_cache_dir / f"{hash_key}.json"
    
    def _get_fetch_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL fetch."""
        normalized = _normalize_url(url)
        hash_key = _hash_key(normalized)
        return self.fetch_cache_dir / f"{hash_key}.json"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """Load data from cache if it exists."""
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict[str, Any]) -> None:
        """Save data to cache."""
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def web_search(self, query: str, num_results: int = 10) -> SearchResponse:

        cache_path = self._get_search_cache_path(query)
        
        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            response = SearchResponse.from_dict(cached_data)
            response.cached = True
            return response
        
        results = self._execute_search(query, num_results)
        
        response = SearchResponse(
            query=query,
            results=results,
            cached=False,
        )
        
        self._save_to_cache(cache_path, response.to_dict())
        
        return response
    
    def _execute_search(self, query: str, num_results: int) -> List[SearchResult]:
        if self.search_engine == "serper":
            return self._search_with_serper(query, num_results)
        else:
            raise ValueError(f"Unknown search engine: {self.search_engine}")
    
    def _search_with_serper(self, query: str, num_results: int) -> List[SearchResult]:
        if not self.search_api_key:
            raise RuntimeError(
                "SERPER_API_KEY not set. Set it in environment or pass search_api_key."
            )
        
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.search_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "num": num_results,
        }
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.request_timeout,
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Serper API error: {response.status_code} - {response.text}"
            )
        
        data = response.json()
        results: List[SearchResult] = []
        
        for rank, item in enumerate(data.get("organic", [])[:num_results], start=1):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                rank=rank,
            ))
        
        return results
    
    def web_fetch(self, url: str) -> FetchResult:

        cache_path = self._get_fetch_cache_path(url)
        
        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            result = FetchResult.from_dict(cached_data)
            result.cached = True
            return result
        
        result = self._execute_fetch(url)
        
        self._save_to_cache(cache_path, result.to_dict())
        
        return result
    
    def _execute_fetch(self, url: str) -> FetchResult:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=self.request_timeout,
                allow_redirects=True,
            )
            
            html = response.text
            text_content = _extract_text_from_html(html)
            title = _extract_title_from_html(html)
            
            return FetchResult(
                url=url,
                final_url=response.url,
                status_code=response.status_code,
                title=title,
                text_content=text_content,
                cached=False,
                error=None,
            )
            
        except requests.RequestException as e:
            return FetchResult(
                url=url,
                final_url=url,
                status_code=0,
                title="",
                text_content="",
                cached=False,
                error=str(e),
            )
   
    def get_cache_stats(self) -> Dict[str, int]:
        search_count = len(list(self.search_cache_dir.glob("*.json")))
        fetch_count = len(list(self.fetch_cache_dir.glob("*.json")))
        return {
            "search_cached": search_count,
            "fetch_cached": fetch_count,
            "total_cached": search_count + fetch_count,
        }
