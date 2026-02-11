"""网页抓取与媒体下载脚本（中文注释版）。

功能概览：
- 基于 BFS 的页面遍历抓取；
- HTML 清洗与正文提取；
- 图片/视频资源筛选下载；
- 可选 m3u8 -> mp4 转码（依赖 ffmpeg）；
- 输出结构化 JSONL / 文件目录。

注意：
- 本脚本包含 robots 与域名策略相关逻辑，使用前请确认合规性。
"""

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Set, Tuple, Dict
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# ASCII-only file. Keep messages concise and neutral.
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from Crypto.Cipher import AES as CryptoAES
except Exception:
    try:
        from Cryptodome.Cipher import AES as CryptoAES
    except Exception:
        CryptoAES = None

AD_KEYWORDS = (
    "ads", "advert", "advertisement", "banner", "sponsor", "promo",
    "promotion", "affiliate", "tracking", "doubleclick",
)

MEDIA_SKIP_KEYWORDS = (
    "icon", "logo", "sprite", "avatar", "emoji", "favicon", "thumb",
    "thumbnail", "small", "btn", "button", "badge", "ads", "advert",
    "promo", "banner", "pixel", "tracking",
)

ENCRYPTED_IMAGE_ATTRS = ("data-xkrkllgl",)

PLACEHOLDER_KEYS = (
    "/assets/place/",
    "/assets/images/",
    "51hl_zw.png",
    "51hl_banner.png",
    "51_icon.png",
    "ads-close.png",
)

LAYOUT_SKIP_KEYWORDS = (
    "nav", "menu", "footer", "header", "sidebar", "aside", "banner",
    "breadcrumb", "crumb", "pager", "pagination", "ads", "advert", "promo",
)

SKIP_EXTS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".avif",
    ".svg", ".ico", ".css", ".js", ".pdf", ".zip", ".rar", ".7z",
    ".mp4", ".m3u8", ".webm", ".mov", ".mkv", ".avi", ".mp3",
    ".woff", ".woff2", ".ttf", ".otf",
)

IMAGE_ATTRS = (
    "data-xkrkllgl",
    "data-src", "data-original", "data-lazy", "data-url", "data-img",
    "data-image", "data-href", "src",
)

LAZY_IMAGE_ATTRS = IMAGE_ATTRS[:-1]

VIDEO_EXTS = (".mp4", ".m3u8", ".webm", ".mov", ".mkv", ".avi")

PAGINATION_TEXTS = (
    "next", "older", "more", "page", "pagenext", "newer", ">>", ">"
)

PAGINATION_CLASS_KEYS = ("next", "pagination", "pager", "page")

INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
MAX_NAME_LEN = 120
LOG_EVERY = 10
MIN_IMAGE_DIM = 120

ENCRYPTED_IMAGE_KEY = b"97b60394abc2fbe1"
ENCRYPTED_IMAGE_IV = b"f5d965df75336270"

CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/avif": ".avif",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/avi": ".avi",
    "application/vnd.apple.mpegurl": ".m3u8",
    "application/x-mpegurl": ".m3u8",
    "audio/mpegurl": ".m3u8",
}


@dataclass
class CrawlConfig:
    """抓取器运行参数集合。"""
    base_url: str
    output_path: str
    max_pages: int
    delay: float
    timeout: float
    retries: int
    user_agent: str
    respect_robots: bool
    allow_subdomains: bool
    download_dir: str
    download_media: bool
    allow_external_media: bool
    min_image_bytes: int
    min_video_bytes: int
    download_mp4: bool
    ffmpeg_path: str


class SimpleCrawler:
    """简化网页抓取器：负责抓取、解析、过滤与资源下载。"""
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
        })
        self.base_domain = urlparse(config.base_url).netloc
        self.allowed_page_domains: Set[str] = {self.base_domain}
        self.allowed_media_domains: Set[str] = {self.base_domain}
        self.visited: Set[str] = set()
        self.queued: Set[str] = set()
        self.title_counts: Dict[str, int] = {}
        self.used_dirs: Set[str] = set()
        self.ffmpeg_path = self._find_ffmpeg()
        self.ffmpeg_warned = False
        self.crypto_warned = False
        self.robots = self._init_robots() if config.respect_robots else None

    def _init_robots(self) -> Optional[RobotFileParser]:
        robots_url = urljoin(self.config.base_url, "/robots.txt")
        try:
            resp = self.session.get(robots_url, timeout=min(self.config.timeout, 10))
        except requests.RequestException:
            return None
        if resp.status_code >= 400:
            return None
        content_type = resp.headers.get("Content-Type", "").lower()
        text = resp.text or ""
        if "text/plain" not in content_type:
            head = text.lstrip()[:200].lower()
            if head.startswith("<!doctype") or head.startswith("<html"):
                return None
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.parse(text.splitlines())
        return rp

    def _find_ffmpeg(self) -> Optional[str]:
        if self.config.ffmpeg_path:
            if os.path.isfile(self.config.ffmpeg_path):
                return self.config.ffmpeg_path
            found = shutil.which(self.config.ffmpeg_path)
            if found:
                return found
        return shutil.which("ffmpeg")

    def _log_ffmpeg_failure(self, dest_path: str, stderr: str) -> None:
        if not dest_path:
            return
        log_path = dest_path + ".ffmpeg.log"
        try:
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write(stderr or "ffmpeg failed with empty stderr\n")
        except OSError:
            pass

    def _allowed_by_robots(self, url: str) -> bool:
        if not self.robots:
            return True
        try:
            allowed = self.robots.can_fetch(self.config.user_agent, url)
            if allowed:
                return True
            if not getattr(self.robots, "entries", None) and getattr(self.robots, "default_entry", None) is None:
                return True
            return False
        except Exception:
            return True

    def _normalize_url(self, url: str, base: str) -> Optional[str]:
        if not url:
            return None
        url = url.strip()
        if url.startswith("#"):
            return None
        if url.lower().startswith(("mailto:", "javascript:", "tel:", "data:")):
            return None
        absolute = urljoin(base, url)
        absolute, _ = urldefrag(absolute)
        return absolute

    def _register_media_domain(self, base_url: Optional[str]) -> None:
        if not base_url:
            return
        parsed = urlparse(base_url)
        if parsed.netloc:
            self.allowed_media_domains.add(parsed.netloc)

    def _register_page_domain(self, base_url: Optional[str]) -> None:
        if not base_url:
            return
        parsed = urlparse(base_url)
        if parsed.netloc:
            self.allowed_page_domains.add(parsed.netloc)

    def _register_page_domains_from_soup(self, soup: BeautifulSoup, base_url: str) -> None:
        if soup is None:
            return
        canonical = soup.find("link", rel=lambda v: v and "canonical" in v)
        if canonical and canonical.get("href"):
            self._register_page_domain(self._normalize_url(canonical.get("href"), base_url))
        og_url = soup.find("meta", attrs={"property": "og:url"})
        if og_url and og_url.get("content"):
            self._register_page_domain(self._normalize_url(og_url.get("content"), base_url))

    def _extract_cdn_base(self, html_text: str) -> Optional[str]:
        if not html_text:
            return None
        patterns = [
            r"DEFAULT_CDN\s*:\s*\"([^\"]+)\"",
            r"DEFAULT_CDN\s*=\s*\"([^\"]+)\"",
            r"DEFAULT_CDN\s*:\s*'([^']+)'",
            r"DEFAULT_CDN\s*=\s*'([^']+)'",
        ]
        for pattern in patterns:
            match = re.search(pattern, html_text)
            if match:
                value = match.group(1)
                value = value.replace("\\/", "/")
                return value.strip()
        return None

    def _element_uses_cdn(self, element) -> bool:
        if element is None:
            return False
        if element.has_attr("data-cdn"):
            return True
        classes = element.get("class", [])
        return "data-cdn" in classes

    def _normalize_media_url(self, url: str, base: str, cdn_base: Optional[str], element) -> Optional[str]:
        if not url:
            return None
        url = url.strip()
        if cdn_base and self._element_uses_cdn(element):
            if not url.lower().startswith("http"):
                return urljoin(cdn_base.rstrip("/") + "/", url.lstrip("/"))
        return self._normalize_url(url, base)

    def _has_lazy_source(self, img) -> bool:
        for attr in LAZY_IMAGE_ATTRS:
            value = img.get(attr)
            if value:
                return True
        return False

    def _looks_like_placeholder(self, url: str) -> bool:
        lower = url.lower()
        for key in PLACEHOLDER_KEYS:
            if key in lower:
                return True
        return False

    def _is_internal(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return False
        if self.config.allow_subdomains:
            for domain in self.allowed_page_domains:
                if parsed.netloc == domain or parsed.netloc.endswith(f".{domain}"):
                    return True
            return False
        return parsed.netloc in self.allowed_page_domains

    def _is_media_domain_allowed(self, url: str) -> bool:
        if self.config.allow_external_media:
            return True
        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return False
        if self.config.allow_subdomains:
            if parsed.netloc.endswith(self.base_domain):
                return True
            return any(parsed.netloc.endswith(domain) for domain in self.allowed_media_domains)
        return parsed.netloc in self.allowed_media_domains

    def _looks_like_ad(self, url: str, element) -> bool:
        if not url:
            return True
        if element is not None:
            classes = element.get("class", [])
            if "tjtagmanager" in classes or "track-click" in classes:
                return True
            if element.name == "a":
                rel = element.get("rel", [])
                if isinstance(rel, str):
                    rel = [rel]
                if any(r.lower() in ("nofollow", "sponsored") for r in rel):
                    return True
            if element.has_attr("data-event") and "ad_click" in str(element.get("data-event")):
                return True
            for attr_key in element.attrs.keys():
                if str(attr_key).lower().startswith("data-ad"):
                    return True
        parsed = urlparse(url)
        haystack = f"{parsed.netloc}{parsed.path}".lower()
        for keyword in AD_KEYWORDS:
            if keyword.isalnum() and len(keyword) <= 3:
                if re.search(rf"(^|[^a-z0-9]){re.escape(keyword)}([^a-z0-9]|$)", haystack):
                    return True
            else:
                if keyword in haystack:
                    return True
        if element is None:
            return False
        # Check a few ancestor levels for ad-related class/id.
        current = element
        for _ in range(3):
            if current is None:
                break
            attrs = " ".join(filter(None, [
                " ".join(current.get("class", [])) if current.has_attr("class") else "",
                current.get("id", "") if current.has_attr("id") else "",
            ])).lower()
            if any(k in attrs for k in AD_KEYWORDS):
                return True
            current = current.parent
        return False

    def _is_ad_container(self, element) -> bool:
        current = element
        for _ in range(5):
            if current is None:
                break
            if current.name == "a":
                rel = current.get("rel", [])
                if isinstance(rel, str):
                    rel = [rel]
                if any(r.lower() in ("nofollow", "sponsored") for r in rel):
                    return True
                classes = current.get("class", [])
                if "track-click" in classes or "tjtagmanager" in classes:
                    return True
                if current.has_attr("data-id"):
                    return True
                if current.has_attr("data-event") and "ad_click" in str(current.get("data-event")):
                    return True
                for attr_key in current.attrs.keys():
                    if str(attr_key).lower().startswith("data-ad"):
                        return True
            classes = " ".join(current.get("class", [])).lower()
            if any(k in classes for k in ("ad", "ads", "banner", "promo", "sponsor", "track-click", "tjtagmanager")):
                return True
            current = current.parent
        return False

    def _looks_like_icon(self, url: str, element) -> bool:
        if not url:
            return True
        lower = url.lower()
        if any(k in lower for k in MEDIA_SKIP_KEYWORDS):
            return True
        if element is None:
            return False
        attrs = " ".join(filter(None, [
            " ".join(element.get("class", [])) if element.has_attr("class") else "",
            element.get("id", "") if element.has_attr("id") else "",
            element.get("alt", "") if element.has_attr("alt") else "",
            element.get("title", "") if element.has_attr("title") else "",
        ])).lower()
        if any(k in attrs for k in MEDIA_SKIP_KEYWORDS):
            return True
        width = element.get("width")
        height = element.get("height")
        try:
            w = int(re.sub(r"\\D", "", str(width))) if width is not None else None
            h = int(re.sub(r"\\D", "", str(height))) if height is not None else None
        except ValueError:
            w = None
            h = None
        if w is not None and h is not None and w <= MIN_IMAGE_DIM and h <= MIN_IMAGE_DIM:
            return True
        return False

    def _is_pagination_link(self, a_tag, href: str) -> bool:
        text = (a_tag.get_text() or "").strip().lower()
        rel = a_tag.get("rel", [])
        if isinstance(rel, str):
            rel = [rel]
        if any(r == "next" for r in rel):
            return True
        if any(t in text for t in PAGINATION_TEXTS):
            return True
        classes = " ".join(a_tag.get("class", [])).lower()
        if any(k in classes for k in PAGINATION_CLASS_KEYS):
            return True
        for idx, parent in enumerate(a_tag.parents):
            if idx >= 4:
                break
            if not getattr(parent, "get", None):
                continue
            parent_classes = " ".join(parent.get("class", [])).lower()
            if any(k in parent_classes for k in ("page-nav", "page-navigator", "pagination", "pager", "next", "prev")):
                return True
        if re.fullmatch(r"\d{1,4}", text) and re.search(r"page|index|list|thread", href, re.I):
            return True
        if re.search(r"page=\d+|/page/\d+|index_\d+\.html", href, re.I):
            return True
        return False

    def _sanitize_component(self, text: str) -> str:
        text = (text or "").strip()
        text = INVALID_PATH_CHARS_RE.sub("_", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" .")
        if not text:
            return "untitled"
        if len(text) > MAX_NAME_LEN:
            text = text[:MAX_NAME_LEN].rstrip(" .")
            if not text:
                return "untitled"
        return text

    def _title_from_url(self, url: str) -> str:
        path = urlparse(url).path.rstrip("/")
        slug = os.path.basename(path)
        return slug or "untitled"

    def _prepare_page_dir(self, title: str, url: str) -> str:
        base = self._sanitize_component(title) if title else self._sanitize_component(self._title_from_url(url))
        count = self.title_counts.get(base, 0)
        while True:
            count += 1
            name = base if count == 1 else f"{base}_{count}"
            path = os.path.join(self.config.download_dir, name)
            if path not in self.used_dirs and not os.path.exists(path):
                break
        self.title_counts[base] = count
        self.used_dirs.add(path)
        os.makedirs(path, exist_ok=True)
        return path

    def _unique_file_path(self, dir_path: str, filename: str) -> str:
        base, ext = os.path.splitext(filename)
        candidate = os.path.join(dir_path, filename)
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(dir_path, f"{base}_{counter}{ext}")
            counter += 1
        return candidate

    def _filename_from_url(self, url: str, prefix: str, index: int) -> str:
        path = urlparse(url).path
        name = os.path.basename(path)
        name = self._sanitize_component(name)
        if not name or name == "untitled":
            name = f"{prefix}_{index}"
        return name

    def _ext_from_content_type(self, content_type: str) -> Optional[str]:
        if not content_type:
            return None
        ctype = content_type.split(";", 1)[0].strip().lower()
        return CONTENT_TYPE_EXT.get(ctype)

    def _sniff_extension(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, "rb") as handle:
                header = handle.read(512)
        except OSError:
            return None
        if header.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            return ".gif"
        if header.startswith(b"BM"):
            return ".bmp"
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return ".webp"
        if header.startswith(b"\x1aE\xdf\xa3"):
            return ".webm"
        if header.startswith(b"RIFF") and header[8:12] == b"AVI ":
            return ".avi"
        if header[4:8] == b"ftyp":
            return ".mp4"
        if header.lstrip().startswith(b"#EXTM3U"):
            return ".m3u8"
        return None

    def _sniff_extension_bytes(self, data: bytes) -> Optional[str]:
        if not data:
            return None
        header = data[:512]
        if header.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            return ".gif"
        if header.startswith(b"BM"):
            return ".bmp"
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return ".webp"
        if header.startswith(b"\x1aE\xdf\xa3"):
            return ".webm"
        if header.startswith(b"RIFF") and header[8:12] == b"AVI ":
            return ".avi"
        if header[4:8] == b"ftyp":
            return ".mp4"
        if header.lstrip().startswith(b"#EXTM3U"):
            return ".m3u8"
        return None

    def _pkcs7_unpad(self, data: bytes) -> Optional[bytes]:
        if not data:
            return None
        pad = data[-1]
        if pad < 1 or pad > 16:
            return None
        if data[-pad:] != bytes([pad]) * pad:
            return None
        return data[:-pad]

    def _decrypt_image_bytes(self, data: bytes) -> Optional[bytes]:
        if not data:
            return None
        if CryptoAES is None:
            if not self.crypto_warned:
                print("pycryptodome not installed; encrypted images skipped. Run: pip install pycryptodome")
                self.crypto_warned = True
            return None
        if len(data) % 16 != 0:
            return None
        try:
            cipher = CryptoAES.new(ENCRYPTED_IMAGE_KEY, CryptoAES.MODE_CBC, ENCRYPTED_IMAGE_IV)
            plaintext = cipher.decrypt(data)
        except Exception:
            return None
        return self._pkcs7_unpad(plaintext)

    def _fetch_bytes(self, url: str, referer: str) -> Optional[Tuple[bytes, str]]:
        headers = {}
        if referer:
            headers["Referer"] = referer
        for attempt in range(1, self.config.retries + 1):
            try:
                with self.session.get(url, timeout=self.config.timeout, stream=True, headers=headers) as resp:
                    if resp.status_code >= 500:
                        time.sleep(self.config.delay * attempt)
                        continue
                    if resp.status_code == 429:
                        time.sleep(max(self.config.delay, 2.0) * attempt)
                        continue
                    if resp.status_code >= 400:
                        return None
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" in content_type:
                        return None
                    data = resp.content
                    if not data:
                        return None
                    return data, content_type
            except requests.RequestException:
                time.sleep(self.config.delay * attempt)
        return None

    def _download_encrypted_image(self, url: str, dir_path: str, filename: str, referer: str) -> Optional[str]:
        fetched = self._fetch_bytes(url, referer)
        if not fetched:
            return None
        encrypted_data, content_type = fetched
        decrypted = self._decrypt_image_bytes(encrypted_data)
        if not decrypted:
            return None
        if len(decrypted) < self.config.min_image_bytes:
            return None
        ext = self._sniff_extension_bytes(decrypted) or self._ext_from_content_type(content_type) or ".bin"
        base, _ = os.path.splitext(filename)
        dest_path = self._unique_file_path(dir_path, f"{base}{ext}")
        try:
            with open(dest_path, "wb") as handle:
                handle.write(decrypted)
        except OSError:
            return None
        return dest_path

    def _adjust_path_for_type(self, dest_path: str, content_type: str) -> str:
        dir_path = os.path.dirname(dest_path)
        base = os.path.basename(dest_path)
        root, ext = os.path.splitext(base)
        if ext and ext != ".bin":
            return dest_path
        guessed = self._ext_from_content_type(content_type)
        if guessed:
            return self._unique_file_path(dir_path, f"{root}{guessed}")
        if ext == ".bin":
            return dest_path
        return self._unique_file_path(dir_path, f"{root}.bin")

    def _maybe_rename_by_sniff(self, dest_path: str) -> str:
        dir_path = os.path.dirname(dest_path)
        base = os.path.basename(dest_path)
        root, ext = os.path.splitext(base)
        if ext and ext != ".bin":
            return dest_path
        guessed = self._sniff_extension(dest_path)
        if not guessed:
            return dest_path
        new_path = self._unique_file_path(dir_path, f"{root}{guessed}")
        try:
            os.replace(dest_path, new_path)
            return new_path
        except OSError:
            return dest_path

    def _download_file(self, url: str, dest_path: str, referer: str, min_bytes: int) -> Optional[str]:
        headers = {}
        if referer:
            headers["Referer"] = referer
        for attempt in range(1, self.config.retries + 1):
            try:
                with self.session.get(url, timeout=self.config.timeout, stream=True, headers=headers) as resp:
                    if resp.status_code >= 500:
                        time.sleep(self.config.delay * attempt)
                        continue
                    if resp.status_code == 429:
                        time.sleep(max(self.config.delay, 2.0) * attempt)
                        continue
                    if resp.status_code >= 400:
                        return None
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" in content_type:
                        return None
                    effective_min = min_bytes
                    if ".m3u8" in url.lower():
                        effective_min = 0
                    if self._ext_from_content_type(content_type) == ".m3u8":
                        effective_min = 0
                    content_len = resp.headers.get("Content-Length")
                    if content_len:
                        try:
                            if int(content_len) < effective_min:
                                return None
                        except ValueError:
                            pass
                    dest_path = self._adjust_path_for_type(dest_path, content_type)
                    with open(dest_path, "wb") as out_file:
                        for chunk in resp.iter_content(chunk_size=262144):
                            if chunk:
                                out_file.write(chunk)
                    try:
                        if os.path.getsize(dest_path) < effective_min:
                            os.remove(dest_path)
                            return None
                    except OSError:
                        return None
                    return self._maybe_rename_by_sniff(dest_path)
            except requests.RequestException:
                time.sleep(self.config.delay * attempt)
        return None

    def _download_m3u8_to_mp4(self, url: str, dest_path: str, referer: str) -> Optional[str]:
        if not self.ffmpeg_path:
            if not self.ffmpeg_warned:
                print("ffmpeg not found; skip m3u8 -> mp4. Install ffmpeg or use --ffmpeg PATH.")
                self.ffmpeg_warned = True
            return None
        headers = []
        if referer:
            headers.append(f"Referer: {referer}")
        if self.config.user_agent:
            headers.append(f"User-Agent: {self.config.user_agent}")
        header_arg = ""
        if headers:
            header_arg = "\\r\\n".join(headers) + "\\r\\n"
        cmd = [self.ffmpeg_path, "-y", "-loglevel", "error"]
        if header_arg:
            cmd += ["-headers", header_arg]
        cmd += ["-i", url, "-c", "copy", "-bsf:a", "aac_adtstoasc", "-movflags", "+faststart", dest_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except OSError:
            return None
        if result.returncode != 0:
            self._log_ffmpeg_failure(dest_path, result.stderr)
            return None
        try:
            if os.path.getsize(dest_path) == 0:
                os.remove(dest_path)
                return None
        except OSError:
            return None
        return dest_path

    def _download_media(
        self,
        images: List[str],
        videos: List[str],
        encrypted_images: Set[str],
        page_dir: str,
        referer: str,
    ) -> Tuple[List[str], List[str]]:
        image_files: List[str] = []
        video_files: List[str] = []
        items: List[Tuple[str, str]] = [("image", url) for url in images] + [("video", url) for url in videos]
        if not items:
            return image_files, video_files
        iterator: Iterable[Tuple[str, str]]
        if tqdm:
            iterator = tqdm(items, desc="Media", unit="file", leave=False)
        else:
            print(f"Media items: {len(items)} -> {page_dir}")
            iterator = items
        img_idx = 0
        vid_idx = 0
        for kind, url in iterator:
            if kind == "image":
                img_idx += 1
                filename = self._filename_from_url(url, "image", img_idx)
                min_bytes = self.config.min_image_bytes
            else:
                vid_idx += 1
                filename = self._filename_from_url(url, "video", vid_idx)
                min_bytes = self.config.min_video_bytes
            if kind == "image" and url in encrypted_images:
                saved = self._download_encrypted_image(url, page_dir, filename, referer)
            elif kind == "video" and ".m3u8" in url.lower() and self.config.download_mp4:
                root, _ = os.path.splitext(filename)
                if not root:
                    root = f"video_{vid_idx}"
                dest_path = self._unique_file_path(page_dir, f"{root}.mp4")
                saved = self._download_m3u8_to_mp4(url, dest_path, referer)
                if not saved:
                    dest_path = self._unique_file_path(page_dir, filename)
                    saved = self._download_file(url, dest_path, referer, 0)
            else:
                dest_path = self._unique_file_path(page_dir, filename)
                saved = self._download_file(url, dest_path, referer, min_bytes)
            if saved:
                if kind == "image":
                    image_files.append(saved)
                else:
                    video_files.append(saved)
        return image_files, video_files

    def _fetch(self, url: str) -> Optional[str]:
        if not self._allowed_by_robots(url):
            return None
        for attempt in range(1, self.config.retries + 1):
            try:
                resp = self.session.get(url, timeout=self.config.timeout)
                if resp.status_code >= 500:
                    time.sleep(self.config.delay * attempt)
                    continue
                if resp.status_code == 429:
                    time.sleep(max(self.config.delay, 2.0) * attempt)
                    continue
                if resp.status_code >= 400:
                    return None
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    return None
                return resp.text
            except requests.RequestException:
                time.sleep(self.config.delay * attempt)
        return None

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Tuple[List[str], List[str]]:
        pagination_links: List[str] = []
        content_links: List[str] = []
        for a_tag in soup.find_all("a"):
            href_raw = a_tag.get("href")
            href = self._normalize_url(href_raw, base_url)
            if not href:
                continue
            if not self._is_internal(href):
                continue
            if self._looks_like_ad(href, a_tag):
                continue
            if any(urlparse(href).path.lower().endswith(ext) for ext in SKIP_EXTS):
                continue
            if self._is_pagination_link(a_tag, href):
                pagination_links.append(href)
            else:
                content_links.append(href)
        for link_tag in soup.find_all("link", rel=True):
            rel = link_tag.get("rel", [])
            if isinstance(rel, str):
                rel = [rel]
            if "next" not in [r.lower() for r in rel]:
                continue
            href_raw = link_tag.get("href")
            href = self._normalize_url(href_raw, base_url)
            if not href:
                continue
            if not self._is_internal(href):
                continue
            pagination_links.append(href)
        return pagination_links, content_links

    def _extract_title(self, soup: BeautifulSoup) -> str:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            return self._clean_title(og.get("content"))
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return self._clean_title(h1.get_text(strip=True))
        tw = soup.find("meta", attrs={"name": "twitter:title"})
        if tw and tw.get("content"):
            return self._clean_title(tw.get("content"))
        if soup.title and soup.title.get_text(strip=True):
            return self._clean_title(soup.title.get_text(strip=True))
        return ""

    def _clean_title(self, title: str) -> str:
        title = html.unescape(title or "").strip()
        if not title:
            return ""
        lowered = title.lower()
        if "caoliu" in lowered or ".one" in lowered:
            for sep in (" | ", "|", " - ", "-"):
                if sep in title:
                    title = title.split(sep)[0].strip()
                    break
            return title.strip()
        if " | " in title:
            title = title.split(" | ")[0].strip()
        elif "|" in title:
            title = title.split("|")[0].strip()
        if "-" in title:
            parts = title.split("-")
            if len(parts) > 1:
                tail = parts[-1].strip()
                if len(tail) <= 12:
                    title = "-".join(parts[:-1]).strip()
        return title.strip()

    def _select_content_root(self, soup: BeautifulSoup):
        for selector in (".post-content", "article.post", "article", "main", "#post"):
            node = soup.select_one(selector)
            if node is not None:
                return node
        return self._select_content_root_scored(soup)

    def _score_container(self, element) -> int:
        if element is None:
            return 0
        attrs = " ".join(filter(None, [
            " ".join(element.get("class", [])) if element.has_attr("class") else "",
            element.get("id", "") if element.has_attr("id") else "",
        ])).lower()
        if any(k in attrs for k in LAYOUT_SKIP_KEYWORDS):
            return 0
        text_len = len(element.get_text(" ", strip=True))
        media_count = len(element.find_all(["img", "video", "source"]))
        link_count = len(element.find_all("a"))
        if text_len < 80 and media_count == 0:
            return 0
        score = text_len + (media_count * 250) - (link_count * 15)
        if element.find("h1"):
            score += 500
        return score

    def _select_content_root_scored(self, soup: BeautifulSoup):
        best = None
        best_score = 0
        for tag in soup.find_all(["article", "main"]):
            score = self._score_container(tag)
            if score > best_score:
                best = tag
                best_score = score
        if best is not None:
            return best
        for tag in soup.find_all(["section", "div", "td"]):
            score = self._score_container(tag)
            if score > best_score:
                best = tag
                best_score = score
        return best or soup

    def _skip_media(self, url: str, element) -> bool:
        if self._is_ad_container(element):
            return True
        if self._looks_like_ad(url, element):
            return True
        if self._looks_like_placeholder(url):
            return True
        if not self._is_media_domain_allowed(url):
            return True
        if self._looks_like_icon(url, element):
            return True
        return False

    def _extract_media(
        self,
        root,
        soup: BeautifulSoup,
        base_url: str,
        cdn_base: Optional[str],
    ) -> Tuple[List[str], List[str], Set[str]]:
        images: Set[str] = set()
        videos: Set[str] = set()
        encrypted_images: Set[str] = set()

        for img in root.find_all("img"):
            has_lazy = self._has_lazy_source(img)
            for attr in IMAGE_ATTRS:
                src_raw = img.get(attr)
                if not src_raw:
                    continue
                if attr == "src" and has_lazy:
                    continue
                src = self._normalize_media_url(src_raw, base_url, cdn_base, img)
                if not src:
                    continue
                if attr in ENCRYPTED_IMAGE_ATTRS:
                    self._register_media_domain(src)
                if self._skip_media(src, img):
                    continue
                images.add(src)
                if attr in ENCRYPTED_IMAGE_ATTRS:
                    encrypted_images.add(src)
                break

        for video in root.find_all("video"):
            src_raw = video.get("src")
            if src_raw:
                src = self._normalize_media_url(src_raw, base_url, cdn_base, video)
                if src:
                    self._register_media_domain(src)
                if src and not self._skip_media(src, video):
                    videos.add(src)
            for source in video.find_all("source"):
                src_raw = source.get("src")
                if not src_raw:
                    continue
                src = self._normalize_media_url(src_raw, base_url, cdn_base, source)
                if src:
                    self._register_media_domain(src)
                if src and not self._skip_media(src, source):
                    videos.add(src)

        # Some pages link to media directly.
        for a_tag in root.find_all("a"):
            href_raw = a_tag.get("href")
            href = self._normalize_media_url(href_raw, base_url, cdn_base, a_tag)
            if not href:
                continue
            if self._skip_media(href, a_tag):
                continue
            lower = urlparse(href).path.lower()
            if lower.endswith(VIDEO_EXTS):
                self._register_media_domain(href)
                videos.add(href)
            if lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".avif")):
                self._register_media_domain(href)
                images.add(href)

        # DPlayer config may hold video URL and poster.
        for player in soup.find_all(attrs={"data-config": True}):
            config_raw = player.get("data-config")
            if not config_raw:
                continue
            try:
                payload = json.loads(html.unescape(config_raw))
            except json.JSONDecodeError:
                continue
            video_cfg = payload.get("video") if isinstance(payload, dict) else None
            if not isinstance(video_cfg, dict):
                continue
            for key in ("url", "pic", "thumbnails"):
                media_url = video_cfg.get(key)
                if not media_url:
                    continue
                media_url = self._normalize_media_url(media_url, base_url, cdn_base, player)
                if not media_url:
                    continue
                self._register_media_domain(media_url)
                if self._skip_media(media_url, player):
                    continue
                if key == "url":
                    videos.add(media_url)
                else:
                    images.add(media_url)

        return sorted(images), sorted(videos), encrypted_images

    def crawl(self) -> None:
        queue = deque([self.config.base_url])
        self.queued.add(self.config.base_url)
        total_pages = 0
        os.makedirs(os.path.dirname(self.config.output_path) or ".", exist_ok=True)
        if self.config.download_media:
            os.makedirs(self.config.download_dir, exist_ok=True)
            if self.config.download_mp4 and not self.ffmpeg_path:
                print("Warning: ffmpeg not found; mp4 conversion will be skipped.")
        page_bar = None
        if tqdm:
            total = self.config.max_pages if self.config.max_pages > 0 else None
            page_bar = tqdm(total=total, desc="Pages", unit="page")

        with open(self.config.output_path, "w", encoding="utf-8") as out_file:
            while queue:
                if self.config.max_pages > 0 and total_pages >= self.config.max_pages:
                    break
                url = queue.popleft()
                if url in self.visited:
                    continue
                self.visited.add(url)

                html = self._fetch(url)
                time.sleep(self.config.delay)
                if not html:
                    total_pages += 1
                    if page_bar is not None:
                        page_bar.update(1)
                    elif total_pages % LOG_EVERY == 0:
                        print(f"Pages processed: {total_pages}, queue: {len(queue)}")
                    continue

                cdn_base = self._extract_cdn_base(html)
                if cdn_base:
                    self._register_media_domain(cdn_base)
                try:
                    soup = BeautifulSoup(html, "lxml")
                except Exception:
                    soup = BeautifulSoup(html, "html.parser")
                self._register_page_domains_from_soup(soup, url)
                title = self._extract_title(soup)
                content_root = self._select_content_root(soup)
                images, videos, encrypted_images = self._extract_media(content_root, soup, url, cdn_base)
                image_files: List[str] = []
                video_files: List[str] = []
                page_dir = ""
                if self.config.download_media and (images or videos):
                    page_dir = self._prepare_page_dir(title, url)
                    image_files, video_files = self._download_media(
                        images, videos, encrypted_images, page_dir, url
                    )

                record = {
                    "url": url,
                    "title": title,
                    "images": images,
                    "videos": videos,
                    "encrypted_images": sorted(encrypted_images),
                    "image_files": image_files,
                    "video_files": video_files,
                    "page_dir": page_dir,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()

                pagination_links, content_links = self._extract_links(soup, url)
                for link in pagination_links + content_links:
                    if link not in self.queued:
                        self.queued.add(link)
                        queue.append(link)

                total_pages += 1
                if page_bar is not None:
                    page_bar.update(1)
                elif total_pages % LOG_EVERY == 0:
                    print(f"Pages processed: {total_pages}, queue: {len(queue)}")

        if page_bar is not None:
            page_bar.close()
        print(f"Done. Pages processed: {total_pages}")
        print(f"Output: {self.config.output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Crawl a site, collect titles/media, and optionally download files.")
    parser.add_argument("--base", default="https://caoliu.one/", help="Base URL to start.")
    parser.add_argument("--output", default="output.jsonl", help="Output JSONL path.")
    parser.add_argument("--download-dir", default="downloads",
                        help="Directory to store media folders.")
    parser.add_argument("--no-download-media", action="store_true",
                        help="Do not download media files.")
    parser.add_argument("--allow-external-media", action="store_true",
                        help="Allow downloading media from external domains.")
    parser.add_argument("--min-image-bytes", type=int, default=8192,
                        help="Minimum image size in bytes to keep.")
    parser.add_argument("--min-video-bytes", type=int, default=51200,
                        help="Minimum video size in bytes to keep.")
    parser.add_argument("--no-mp4", action="store_true",
                        help="Do not convert m3u8 to mp4.")
    parser.add_argument("--ffmpeg", default="",
                        help="Path to ffmpeg executable.")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="Max pages to crawl (0 = no limit).")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests.")
    parser.add_argument("--timeout", type=float, default=15.0, help="Request timeout.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count.")
    parser.add_argument("--user-agent", default=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
        help="User-Agent string.")
    parser.add_argument("--no-robots", action="store_true",
                        help="Ignore robots.txt (not recommended).")
    parser.add_argument("--allow-subdomains", action="store_true",
                        help="Allow crawling subdomains.")
    return parser


def main() -> None:
    """程序主入口。"""
    parser = build_arg_parser()
    args = parser.parse_args()

    config = CrawlConfig(
        base_url=args.base,
        output_path=args.output,
        max_pages=args.max_pages,
        delay=max(args.delay, 0.0),
        timeout=max(args.timeout, 1.0),
        retries=max(args.retries, 1),
        user_agent=args.user_agent,
        respect_robots=not args.no_robots,
        allow_subdomains=args.allow_subdomains,
        download_dir=args.download_dir,
        download_media=not args.no_download_media,
        allow_external_media=args.allow_external_media,
        min_image_bytes=max(args.min_image_bytes, 0),
        min_video_bytes=max(args.min_video_bytes, 0),
        download_mp4=not args.no_mp4,
        ffmpeg_path=args.ffmpeg,
    )

    crawler = SimpleCrawler(config)
    crawler.crawl()


if __name__ == "__main__":
    main()
