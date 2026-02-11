"""
Configuration management for the telemedicine MCP server.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """
    Application settings loaded from environment variables.
    """

    def __init__(self) -> None:
        # API Configuration
        self.base_url: str = os.getenv("BASE_URL", "")
        if not self.base_url:
            raise ValueError("BASE_URL must be set in .env file")

        # Remove trailing slash if present
        self.base_url = self.base_url.rstrip("/")

        # Server Configuration
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.timeout: int = int(os.getenv("TIMEOUT", "30"))

        # Authentication (placeholder for future implementation)
        self.jwt_secret: Optional[str] = os.getenv("JWT_SECRET")
        self.api_key: Optional[str] = os.getenv("API_KEY")

        # Pagination (placeholder for future implementation)
        self.default_page_size: int = int(os.getenv("DEFAULT_PAGE_SIZE", "20"))
        self.max_page_size: int = int(os.getenv("MAX_PAGE_SIZE", "100"))

    def get_headers(self) -> dict[str, str]:
        """
        Get common headers for API requests.
        
        Returns:
            Dictionary of headers to include in requests.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Future: Add authentication headers
        # if self.api_key:
        #     headers["X-API-Key"] = self.api_key
        # if self.jwt_token:
        #     headers["Authorization"] = f"Bearer {self.jwt_token}"

        return headers


# Global settings instance
settings = Settings()