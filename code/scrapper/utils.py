import re
import base64
import logging

# This module provides utility functions for processing URLs, particularly those containing Base64 encoded segments.

def get_pretty_url(url):
    """
    Extracts a pretty URL from a given URL that contains a Base64 encoded segment.
    If the URL does not contain a Base64 encoded segment, it returns the original URL.
    If the Base64 encoded segment is malformed or decoding fails, it returns the original URL.
    Args:
        url (str): The URL containing a Base64 encoded segment.
    Returns:
        str: A cleaned and decoded URL if successful, otherwise the original URL.
    """
    base64_pattern = r"articles\/([A-Za-z0-9+_\-\/=]+)"
    match = re.search(base64_pattern, url)

    if match and match.group(1):
        base64_encoded_url = match.group(1).replace("-", "+").replace("_", "/")
        # Add padding to the base64_encoded_url
        base64_encoded_url += "=" * ((4 - len(base64_encoded_url) % 4) % 4)
        try:
            decoded_url = base64.urlsafe_b64decode(base64_encoded_url).decode("latin1")

            # Remove any trailing "R" if it's the last character
            decoded_url = decoded_url.rstrip("R")

            # Remove non-ASCII characters and split by potential delimiters
            parts = re.split(r"[^\x20-\x7E]+", decoded_url)

            # Regular expression to validate and extract URLs
            url_pattern = r"(https?:\/\/[^\s]+)"
            cleaned_url = ""

            # Iterate over parts to find the first valid URL
            for part in parts:
                url_match = re.search(url_pattern, part)
                if url_match and url_match.group(1):
                    cleaned_url = url_match.group(1)
                    break  # Stop at the first match

            if cleaned_url:
                return cleaned_url
            else:
                logging.error(f"No valid URL found in the decoded string: {decoded_url}")
                return url
        except UnicodeDecodeError as error:
            logging.error(f"Error decoding Base64 string: {base64_encoded_url}, \
                          Original URL: {url}, Error: {error}")
            return url
    else:
        logging.error(f"No Base64 segment found in the URL. Original URL: {url}")
        return url
