import re


class TextPreprocessor:
    """
    Handles cleaning and normalization of raw text inputs
    before they are tokenized and processed by the transformer model.
    """

    def __init__(self, lowercase: bool = False, remove_extra_spaces: bool = True):
        self.lowercase = lowercase
        self.remove_extra_spaces = remove_extra_spaces

    def clean(self, text: str) -> str:
        """
        Cleans and normalizes the input text.
        
        Args:
            text: Raw input text.
            
        Returns:
            Cleaned and normalized text.
        """
        if not text:
            return ""

        # Remove control characters and non-printable characters
        cleaned = "".join(ch for ch in text if ch.isprintable() or ch in ("\n", "\r", "\t"))

        # Strip leading/trailing whitespaces
        cleaned = cleaned.strip()

        # Conditionally convert to lowercase (note: DistilRoBERTa is typically case-sensitive,
        # but we allow this as a configurable parameter if needed)
        if self.lowercase:
            cleaned = cleaned.lower()

        # Replace multiple spaces/tabs/newlines with a single space if configured
        if self.remove_extra_spaces:
            cleaned = re.sub(r"\s+", " ", cleaned)

        return cleaned
