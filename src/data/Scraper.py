from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests import Session
from requests.exceptions import RequestException, Timeout


BASE_URL = "http://books.toscrape.com/"
OUTPUT_FILE = Path("data/raw/books_raw_data.csv")

REQUEST_TIMEOUT = 10
REQUEST_DELAY = 0.5

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def create_session() -> Session:
    """Create and configure a reusable HTTP session."""
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/139.0 Safari/537.36"
            )
        }
    )

    return session


def fetch_page(session: Session, url: str) -> str | None:
    """
    Download a webpage.

    Returns:
        HTML content if successful, otherwise None.
    """
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text

    except Timeout:
        print(f"[WARNING] Timeout while requesting: {url}")
        return None

    except RequestException as exc:
        print(f"[WARNING] Request failed for {url}: {exc}")
        return None


def clean_text(text: str | None) -> str:
    """Normalize whitespace in scraped text."""
    if not text:
        return ""

    return " ".join(text.split())


def parse_price(price_text: str | None) -> float | None:
    """
    Convert a price string such as '£51.77' into 51.77.
    """
    if not price_text:
        return None

    match = re.search(r"[\d.]+", price_text)

    if not match:
        return None

    try:
        return float(match.group())
    except ValueError:
        return None


def parse_availability(availability_text: str | None) -> int | None:
    """
    Convert availability text such as
    'In stock (19 available)'
    into integer 19.
    """
    if not availability_text:
        return None

    match = re.search(r"\((\d+)\s+available\)", availability_text)

    if not match:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None


def parse_rating(rating_element) -> int | None:
    """
    Convert the CSS rating class into an integer.

    Example:
        class="star-rating Three" -> 3
    """
    if rating_element is None:
        return None

    classes = rating_element.get("class", [])

    for class_name in classes:
        if class_name in RATING_MAP:
            return RATING_MAP[class_name]

    return None


def get_category(soup: BeautifulSoup) -> str:
    """Extract the book category from the breadcrumb."""
    breadcrumb = soup.select("ul.breadcrumb li")

    if len(breadcrumb) >= 3:
        return clean_text(breadcrumb[-2].get_text())

    return ""


def get_product_description(soup: BeautifulSoup) -> str:
    """Extract the product description from the product page."""
    description_heading = soup.find(
        "div",
        id="product_description",
    )

    if description_heading:
        description_paragraph = description_heading.find_next_sibling(
            "p"
        )

        if description_paragraph:
            return clean_text(description_paragraph.get_text())

    return ""


def get_upc(soup: BeautifulSoup) -> str:
    """Extract UPC from the product information table."""
    product_table = soup.select_one("table.table-striped")

    if not product_table:
        return ""

    for row in product_table.select("tr"):
        header = row.find("th")
        value = row.find("td")

        if header and value:
            field_name = clean_text(header.get_text())

            if field_name == "UPC":
                return clean_text(value.get_text())
            
    return ""


def parse_product_page(
    session: Session,
    product_url: str,
) -> dict[str, object] | None:
    """
    Scrape one individual product page.
    """
    html = fetch_page(session, product_url)

    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_element = soup.select_one("div.product_main h1")
    title = clean_text(
        title_element.get_text() if title_element else ""
    )

    # Price
    price_element = soup.select_one("div.product_main .price_color")
    price = parse_price(
        price_element.get_text() if price_element else None
    )

    # Availability
    availability_element = soup.select_one(
        "div.product_main .availability"
    )

    availability = parse_availability(
        availability_element.get_text()
        if availability_element
        else None
    )

    # Rating
    rating_element = soup.select_one(
        "div.product_main .star-rating"
    )

    rating = parse_rating(rating_element)

    # Category
    category = get_category(soup)

    # Description
    product_description = get_product_description(soup)

    # UPC
    upc = get_upc(soup)

    return {
        "title": title,
        "price": price,
        "availability": availability,
        "rating": rating,
        "category": category,
        "product_description": product_description,
        "upc": upc,
    }


def get_product_links(
    soup: BeautifulSoup,
) -> list[str]:
    """Extract all product links from one catalogue page."""
    links = []

    for article in soup.select("article.product_pod"):
        link = article.select_one("h3 a")

        if link and link.get("href"):
            links.append(link["href"])

    return links


def make_absolute_url(
    current_url: str,
    relative_url: str,
) -> str:
    """Convert a relative URL into an absolute URL."""
    from urllib.parse import urljoin

    return urljoin(current_url, relative_url)


def get_next_page_url(
    soup: BeautifulSoup,
    current_url: str,
) -> str | None:
    """Find the next catalogue page."""
    next_link = soup.select_one("li.next a")

    if not next_link or not next_link.get("href"):
        return None

    return make_absolute_url(
        current_url,
        next_link["href"],
    )


def scrape_catalogue(
    session: Session,
) -> list[dict[str, object]]:
    """
    Scrape all catalogue pages and all products.
    """
    all_books = []
    current_url = BASE_URL
    page_number = 1

    while current_url:
        print(
            f"[INFO] Scraping catalogue page {page_number}: "
            f"{current_url}"
        )

        html = fetch_page(session, current_url)

        if html is None:
            print(
                f"[WARNING] Skipping catalogue page "
                f"{page_number}."
            )

            break

        soup = BeautifulSoup(html, "html.parser")

        product_links = get_product_links(soup)

        print(
            f"[INFO] Found {len(product_links)} products "
            f"on page {page_number}."
        )

        for index, relative_link in enumerate(product_links, start=1):
            product_url = make_absolute_url(
                current_url,
                relative_link,
            )

            print(
                f"    [PRODUCT {index}/{len(product_links)}] "
                f"{product_url}"
            )

            book = parse_product_page(
                session,
                product_url,
            )

            if book is not None:
                all_books.append(book)

            # Polite delay between requests
            time.sleep(REQUEST_DELAY)

        # Find next catalogue page
        next_url = get_next_page_url(
            soup,
            current_url,
        )

        current_url = next_url
        page_number += 1

        # Polite delay before requesting next catalogue page
        if current_url:
            time.sleep(REQUEST_DELAY)

    return all_books


def save_to_csv(
    books: list[dict[str, object]],
    output_file: Path,
) -> None:
    """Save scraped books into a CSV file."""
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "title",
        "price",
        "availability",
        "rating",
        "category",
        "product_description",
        "upc",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(books)


def main() -> None:
    """Main entry point for the scraper."""
    print("=" * 60)
    print("Books to Scrape - Phase 1 Data Scraper")
    print("=" * 60)

    session = create_session()

    books = scrape_catalogue(session)

    save_to_csv(
        books,
        OUTPUT_FILE,
    )

    print("=" * 60)
    print(
        f"[SUCCESS] Scraping completed. "
        f"{len(books)} books collected."
    )
    print(f"[SUCCESS] Output saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()